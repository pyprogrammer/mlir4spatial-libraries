package tensorflow_lattice

import mlir_libraries.types.Readable2D
import spatial.libdsl._

trait PWLCalibration {

  def PWLCalibration_Sequential[T: Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg: Readable2D[T])(implicit state: argon.State) = {
    // kernel_list(0) is the bias

    val units = {
      val lengths = (pwl_calibration_kernel map {_.length}).distinct
      assert(lengths.length == 1, f"Found multiple possible number of units for PWL Calibration. Expected 1. ${lengths}")
      lengths.head
    }

    val bias = LUT[T](pwl_calibration_kernel.head.length)((pwl_calibration_kernel.head map {x => Bits(x.toUnchecked[T])}):_*)
    // the rest of the kernel list is the actual weights
    val weights = pwl_calibration_kernel.tail
    val weight_luts = weights map {
      vec => {
//        LUT[T](vec.length)((vec map {x => Bits(x.toUnchecked[T])}):_*)
        (vec map {x => x.toUnchecked[T]})
      }
    }

    val keypoint_list = input_keypoints
    // compute a (start, end, delta) list
    val start_end_delta = keypoint_list zip keypoint_list.tail zip weight_luts

    val recip_lengths = start_end_delta map { case ((a, b), _) => 1 / (b - a) }

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val value = arg(d0, d1)
        // The kernel is phrased as deltas, so we add up all deltas less than the value
        val sub_components = start_end_delta zip recip_lengths map { case (((s, _), d), l) =>
          val raw_weight: T = (value - s.toUnchecked[T]) * l.toUnchecked[T]
//          val result: T = max(min(raw_weight, 1), 0) * d(d1)
          val result: T = max(min(raw_weight, 1), 0) * oneHotMux(Seq.tabulate(weights.size){i => i.to[I32] == d1}, d.toSeq)
          result
        }
        sub_components.fold(bias(d1)) {
          _ + _
        }
      }

      lazy val shape: Seq[I32] = Seq(arg.shape(0), I32(units))
    }
  }

  def PWLCalibration_mux[T: Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg: Readable2D[T])(implicit state: argon.State) = {

    // kernel is phrased as bias :: deltas.
    // however, we wish to use a priority mux instead, so we first compute the running sum.

    val units = {
      val lengths = (pwl_calibration_kernel map {_.length}).distinct
      assert(lengths.length == 1, f"Found multiple possible number of units for PWL Calibration. Expected 1. ${lengths}")
      lengths.head
    }

    val cumsum = (pwl_calibration_kernel.transpose map {
      vec => vec.tail.scanLeft(vec.head) {_ + _}
    }).transpose map {
      vec => LUT[T](vec.length)((vec map {x => Bits(x.toUnchecked[T])}):_*)
    }
    // cumsum(0) handles everything before the first keypoint and cumsum(last) handles everything after.

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val value = arg(d0, d1)
        val enables = (input_keypoints map { keypoint => keypoint.toUnchecked[T] < value }) ++ Seq(value >= cumsum.last(d1))
        val middle_values = ((input_keypoints zip input_keypoints.tail) zip (cumsum zip cumsum.tail)) map {
          case ((start, end), (left_val, right_val)) =>
            val reciprocal_length = 1 / (end - start)
            val offset = value - start.toUnchecked[T]
            val scaled_amount = offset * reciprocal_length.toUnchecked[T]
            left_val(d1) + scaled_amount * (right_val(d1) - left_val(d1))
        }
        val values = Seq(cumsum.head(d1)) ++ middle_values ++ Seq(cumsum.last(d1))
        assert(enables.length == values.length, "Enables should be of the same length as the values")
        priorityMux(enables, values)
      }

      lazy val shape: Seq[I32] = Seq(arg.shape(0), units)
    }
  }
}
