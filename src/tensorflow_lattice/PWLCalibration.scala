package tensorflow_lattice

import mlir_libraries.types._
import mlir_libraries.{CoprocessorScope, types, Tensor => MLTensor}
import spatial.libdsl._

class PWLCalibration(val pwl_calibration_kernel: MLTensor[scala.Double], val input_keypoints: MLTensor[scala.Double]) {
  private val id = PWLCalibration.getPWLID()

  private val units = pwl_calibration_kernel.shape(1)
  private val num_keypoints = input_keypoints.shape(0)
  private val pwl_calib_array = pwl_calibration_kernel.to2DSeq
  private val input_keypoints_array = input_keypoints.to1DSeq
  private val cumsum = pwl_calib_array.transpose map {
    vec => vec.tail.scanLeft(vec.head) {_ + _}
  }

  private val diffs = pwl_calib_array.transpose map {
    vec => vec.tail
  }

  private val lengths = (input_keypoints_array zip input_keypoints_array.tail) map {
    case (left, right) => right - left
  }

  private val scaled_diffs = diffs map {
    diff_vec =>
      ((diff_vec zip lengths) map {
        case (diff, length) => diff / length
      }) ++ Seq[scala.Double](0)
  }

  private def isDegenerate(arg: ReadableND[_]): Boolean = {
    arg.shape.last match {
      case argon.Const(c) => c.value == 1
      case _ => false
    }
  }

  @forge.tags.stateful def apply[T: Num: Bits](arg: ReadableND[T]): ReadableND[T] = {
    new ReadableND[T] {
      // batch, unit -> batch, unit

      override def getInterface: types.Interface[T] = {
        val subInterface = arg.getInterface
        new types.Interface[T] {
          def getRealIndex(index: Seq[I32]) = {
            if (isDegenerate(arg)) index.dropRight(1) ++ Seq(I32(0)) else index
          }

          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            subInterface.enq(getRealIndex(index), ens)
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val value = subInterface.deq(getRealIndex(index), ens)
            val d1 = index.last

            val input_kp_LUT = LUT[T](num_keypoints)((input_keypoints_array map {x => Bits(x.toUnchecked[T])}):_*)
            val scaled_diffs_LUT = LUT[T](units, num_keypoints)(scaled_diffs.flatten map {x => Bits(x.toUnchecked[T])}:_*)

            val cumsum_LUT = LUT[T](units, num_keypoints)((cumsum.flatten) map {x => Bits(x.toUnchecked[T])}:_*)

            // Compute proper left and right indices
            val before_first = input_keypoints_array.head.to[T] >= value
            val after_last = input_keypoints_array.last.to[T] <= value

            // Ignore 0th kp in sum, can't have left_kp be the rightmost kp.
            val leftKPIndexComponents: Seq[Bit] = input_keypoints_array.tail.dropRight(1) map {
              kp =>
                kp.toUnchecked[T] <= value
            }

            val leftKP = leftKPIndexComponents map {
              bit => mux(bit, I32(1), I32(0))
            } reduceTree { _ + _ }
            val offset = value - input_kp_LUT(leftKP)
            val scaled_diff = scaled_diffs_LUT(d1, leftKP)
            val interpolated = cumsum_LUT(d1, leftKP) + offset * scaled_diff
            priorityMux(Seq(before_first, after_last, Bit(true)), Seq(cumsum_LUT(d1, I32(0)), cumsum_LUT(d1, I32(num_keypoints - 1)), interpolated))
          }
        }
      }

      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
  }

  def checkArgs() = {

  }
}

object PWLCalibration {

  private var pwlID = -1
  private def getPWLID(): Int = {
    pwlID += 1
    pwlID
  }

  def apply(pwl_calibration_kernel: MLTensor[scala.Double], input_keypoints: MLTensor[scala.Double]) = {
    new PWLCalibration(pwl_calibration_kernel, input_keypoints)
  }
}

trait PWLCalibration2 {

  var pwl_id = 0
  def getPWLId() = {
    val tmp = pwl_id
    pwl_id += 1
    tmp
  }

  def PWLCalibration[T: Num : Bits](pwl_calibration_kernel: MLTensor[scala.Double], input_keypoints: MLTensor[scala.Double])(arg: ReadableND[T])
                                   (implicit state: argon.State) = {
    // kernel is phrased as bias :: deltas.
    // however, we wish to use a priority mux instead, so we first compute the running sum.
    val num_loops = 1
    val id = getPWLId()

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

    val lengths = (input_keypoints_array zip input_keypoints_array.tail) map {
      case (left, right) => right - left
    }

    val scaled_diffs = diffs map {
      diff_vec =>
        ((diff_vec zip lengths) map {
          case (diff, length) => diff / length
        }) ++ Seq[scala.Double](0)
    }

    // cumsum(0) handles everything before the first keypoint and cumsum(last) handles everything after.

    val degenerate_PWL_input = arg.shape(1) match {
      case argon.Const(c) => c.value == 1
      case _ => false
    }

    val iterations = num_keypoints - 1

    val par_factor = iterations / num_loops + (if (iterations % num_loops > 0) 1 else 0)

    new ReadableND[T] {
      // batch, unit -> batch, unit

      override def getInterface: types.Interface[T] = {
        val subInterface = arg.getInterface
        new types.Interface[T] {
          def getRealIndex(index: Seq[I32]) = {
            if (degenerate_PWL_input) index.dropRight(1) ++ Seq(I32(0)) else index
          }

          override def enq(index: Seq[I32], ens: Set[Bit]): Void = {
            subInterface.enq(getRealIndex(index), ens)
          }

          override def deq(index: Seq[I32], ens: Set[Bit]): T = {
            val value = subInterface.deq(getRealIndex(index), ens)
            val d1 = index.last

            val input_kp_LUT = LUT[T](num_keypoints)((input_keypoints_array map {x => Bits(x.toUnchecked[T])}):_*)
            val scaled_diffs_LUT = LUT[T](units, num_keypoints)(scaled_diffs.flatten map {x => Bits(x.toUnchecked[T])}:_*)

            val cumsum_LUT = LUT[T](units, num_keypoints)((cumsum.flatten) map {x => Bits(x.toUnchecked[T])}:_*)
            val front_LUT = LUT[T](units, num_keypoints)((cumsum.flatten) map {x => Bits(x.toUnchecked[T])}:_*)

            // Handles cases where the input is within the keypoints.
            // Still need to handle the cases where the input is less than the first keypoint or larger than the last keypoint.
            if (num_loops != 1) {
              val pwl = Pipe.Reduce(Reg[T])(iterations by 1 par I32(par_factor)) {
                kp_index =>
                  val input_kp = input_kp_LUT(kp_index)
                  val next_kp = input_kp_LUT(kp_index + I32(1))
                  val scaled_diff = scaled_diffs_LUT(d1, kp_index)
                  val offset = value - input_kp
                  val is_valid = (input_kp < value) && (value <= next_kp)
                  val output = cumsum_LUT(d1, kp_index) + offset * scaled_diff
                  mux(is_valid, output, 0.to[T])
              } {
                _ + _
              }

              // Handle Edge Cases
              val front_val = front_LUT(d1, I32(0))
              val back_val = front_LUT(d1, I32(num_keypoints - 1))
              val before_first = input_keypoints_array.head.toUnchecked[T] >= value
              val after_last = input_keypoints_array.last.toUnchecked[T] <= value
              priorityMux(Seq[Bit](before_first, after_last, Bit(true)), Seq[T](front_val, back_val, pwl))
            } else {
              val conditions = Seq(input_keypoints_array.head.toUnchecked[T] >= value) ++ (input_keypoints_array map {
                x =>
                  x.toUnchecked[T] < value
              })
              val intermediateInterpolants = Range(0, num_keypoints - 1) map {
                kpIndex =>
                  val input_kp = input_kp_LUT(I32(kpIndex))
                  val scaled_diff = scaled_diffs_LUT(d1, I32(kpIndex))
                  val offset = value - input_kp
                  cumsum_LUT(d1, I32(kpIndex)) + offset * scaled_diff
              }
              val interpolated = Seq(front_LUT(d1, I32(0))) ++ intermediateInterpolants ++ Seq(front_LUT(d1, I32(num_keypoints - 1)))
              assert(conditions.length == interpolated.length, s"Found ${conditions.length} conditions and ${interpolated.length} interpolation results!")
              priorityMux(conditions.reverse, interpolated.reverse)
            }
          }
        }
      }

      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(units))
    }
  }
}
