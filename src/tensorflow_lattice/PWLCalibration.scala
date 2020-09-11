package tensorflow_lattice

import mlir_libraries.types.ReadableND
import mlir_libraries.{Tensor => MLTensor}
import spatial.libdsl._

trait PWLCalibration {
  def PWLCalibration[T: Num : Bits](pwl_calibration_kernel: MLTensor[scala.Double], input_keypoints: MLTensor[scala.Double])(arg: ReadableND[T])(implicit state: argon.State, config: mlir_libraries.OptimizationConfig) = {
    // kernel is phrased as bias :: deltas.
    // however, we wish to use a priority mux instead, so we first compute the running sum.
    val num_loops = config.pwl_iterations

    assert(pwl_calibration_kernel.rank == 2, "PWL Kernel must be rank 2")
    assert(input_keypoints.rank == 1, "Input keypoints must be rank 1")

    val units = pwl_calibration_kernel.shape(1)

    val num_keypoints = input_keypoints.shape(0)

    val pwl_calib_array = pwl_calibration_kernel.to2DSeq
    val input_keypoints_array = input_keypoints.to1DSeq
    // contains output(0), output(1), etc.
    val cumsum = pwl_calib_array.transpose map {
      vec => vec.tail.scanLeft(vec.head) {_ + _}
    }

    val diffs = pwl_calib_array.transpose map {
      vec => vec.tail
    }

    val cumsum_LUT = LUT[T](units, num_keypoints)((cumsum.flatten) map {x => Bits(x.toUnchecked[T])}:_*)

    val input_kp_LUT = LUT[T](num_keypoints)((input_keypoints_array map {x => Bits(x.toUnchecked[T])}):_*)

    val lengths = (input_keypoints_array zip input_keypoints_array.tail) map {
      case (left, right) => right - left
    }

    val scaled_diffs = diffs map {
      diff_vec =>
        ((diff_vec zip lengths) map {
          case (diff, length) => diff / length
        }) ++ Seq[scala.Double](0)
    }

    val scaled_diffs_LUT = LUT[T](units, num_keypoints)(scaled_diffs.flatten map {x => Bits(x.toUnchecked[T])}:_*)

    // cumsum(0) handles everything before the first keypoint and cumsum(last) handles everything after.

    val degenerate_PWL_input = arg.shape(1) match {
      case argon.Const(c) => c.value == 1
      case _ => false
    }

    val iterations = num_keypoints - 1

    val par_factor = iterations / num_loops + (if (iterations % num_loops > 0) 1 else 0)

    new ReadableND[T] {
      // batch, unit -> batch, unit
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val d0 = index.head
        val d1 = index.last
        val out = Reg[T](0).conflictable
        val staged = if (degenerate_PWL_input) arg(Seq(index.head, I32(0)), ens) else arg(index, ens)
        val value = staged()

        {
          import spatial.dsl._
          println(r"PWL Input ($d0: ${d0.ctx}, $d1: ${d1.ctx}): $value")
        }

        Parallel {
          // Handles cases where the input is within the keypoints.
          // Still need to handle the cases where the input is less than the first keypoint or larger than the last keypoint.
          Pipe.Foreach(iterations by 1 par I32(par_factor)) {
            kp_index =>
              val input_kp = input_kp_LUT(kp_index)
              val next_kp = input_kp_LUT(kp_index + I32(1))
              val scaled_diff = scaled_diffs_LUT(d1, kp_index)
              val offset = value - input_kp
              val is_valid = (input_kp < value) && (value <= next_kp)
              val output = cumsum_LUT(d1, kp_index) + offset * scaled_diff
              out.write(output, is_valid)
          }
          // Handle Edge Cases
          Pipe {
            val before_first = input_keypoints_array.head.toUnchecked[T] >= value
            out.write(cumsum_LUT(d1, I32(0)), before_first)

            {
              import spatial.dsl._
              println(r"Before First: $before_first => ${cumsum_LUT(d1, I32(0))}")
            }
          }

          Pipe {
            val after_last = input_keypoints_array.last.toUnchecked[T] <= value
            out.write(cumsum_LUT(d1, I32(num_keypoints - 1)), after_last)

            {
              import spatial.dsl._
              println(r"After Last: $after_last => ${cumsum_LUT(d1, I32(num_keypoints - 1))}")
            }
          }
        }


        {
          import spatial.dsl._
          println(r"Output: $out")
        }

        () => out.value
      }

      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
  }
}
