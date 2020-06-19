package tensorflow_lattice

import mlir_libraries.types.Readable2D
import spatial.libdsl._

trait PWLCalibration {
  def PWLCalibration[T: Num : Bits](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double], num_loops: Int = 1)(arg: Readable2D[T])(implicit state: argon.State) = {
    // kernel is phrased as bias :: deltas.
    // however, we wish to use a priority mux instead, so we first compute the running sum.

    val units = {
      val lengths = (pwl_calibration_kernel map {_.length}).distinct
      assert(lengths.length == 1, f"Found multiple possible number of units for PWL Calibration. Expected 1. ${lengths}")
      lengths.head
    }

    val num_keypoints = input_keypoints.length

    // contains output(0), output(1), etc.
    val cumsum = pwl_calibration_kernel.transpose map {
      vec => vec.tail.scanLeft(vec.head) {_ + _}
    }

    val diffs = pwl_calibration_kernel.transpose map {
      vec => vec.tail
    }

    val cumsum_LUT = LUT[T](units, num_keypoints)((cumsum.flatten) map {x => Bits(x.toUnchecked[T])}:_*)

    val input_kp_LUT = LUT[T](num_keypoints + 1)(((input_keypoints :+ input_keypoints.last) map {x => Bits(x.toUnchecked[T])}):_*)
    val lengths = (input_keypoints zip input_keypoints.tail) map {
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

    val par_factor = num_keypoints / num_loops + (if (num_keypoints % num_loops > 0) 1 else 0)

//    println(s"Is Degenerate: $degenerate_PWL_input")

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): () => T = {
        val out = Reg[T](0).conflictable
        Foreach(num_keypoints by 1 par I32(par_factor)) {
          kp_index =>
            val staged = if (degenerate_PWL_input) arg(d0, I32(0)) else arg(d0, d1)
            val value = staged()
            val input_kp = input_kp_LUT(kp_index)
            val next_kp = input_kp_LUT(kp_index + I32(1))
            val scaled_diff = scaled_diffs_LUT(d0, kp_index)
            val offset = value - input_kp
            val is_valid = (input_kp < value) && (value <= next_kp)
            // Relies on the fact that 0 is a vector of all 0's in type T.
            out.write(cumsum_LUT(d1, kp_index) + offset * scaled_diff, is_valid)
        }

        () => out.value
      }

      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
  }
}
