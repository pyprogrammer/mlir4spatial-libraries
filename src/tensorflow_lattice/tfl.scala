package tensorflow_lattice

import mlir_libraries.types._

import spatial.libdsl._

object tfl {

  private val PWL_mode = sys.env.getOrElse("PWL", "mux") == "mux"
  private val PO2Opt = sys.env.getOrElse("PO2Opt", "true") == "true"

  println("PWL_mode=" + PWL_mode)
  println("PO2Opt=" + PO2Opt)

  def CategoricalCalibration[T : Num](categorical_calibration_kernel: scala.Array[scala.Array[scala.Double]])(arg:Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    val param_list = categorical_calibration_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }.toSeq
    val params = LUT[T](param_list.length)(param_list:_*)
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val value = arg(d0, d1)
        params(value.to[I32])
      }

      lazy val shape: Seq[I32] = arg.shape
    }
  }

  def PWLCalibration_Sequential[T:Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg:Readable2D[T])(implicit state: argon.State) = {
    val kernel_list = pwl_calibration_kernel.flatten
    // kernel_list(0) is the bias
    val bias: T = kernel_list.head.toUnchecked[T]
    // the rest of the kernel list is the actual weights
    val weights = kernel_list.tail

    val keypoint_list = input_keypoints
    // compute a (start, end, delta) list
    val start_end_delta = keypoint_list zip keypoint_list.tail zip weights

    val recip_lengths = start_end_delta map { case ((a, b), _) => 1/(b - a) }

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val value = arg(d0, d1)
        // The kernel is phrased as deltas, so we add up all deltas less than the value
        val sub_components = start_end_delta zip recip_lengths map { case (((s, _), d), l) =>
          val raw_weight: T = (value - s.toUnchecked[T]) * l.toUnchecked[T]
          val result: T = max(min(raw_weight, 1), 0) * d.toUnchecked[T]
          result
        }
        sub_components.fold(bias) {
          _ + _
        }
      }

      lazy val shape: Seq[I32] = arg.shape
    }
  }

  def PWLCalibration_mux[T:Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg:Readable2D[T])(implicit state: argon.State) = {
    val kernel_list = pwl_calibration_kernel.flatten

    // kernel is phrased as bias :: deltas.
    // however, we wish to use a priority mux instead, so we first compute the running sum.
    val cumsum = kernel_list.tail.scanLeft(kernel_list.head){_ + _}
    // cumsum(0) handles everything before the first keypoint and cumsum(last) handles everything after.

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val value = arg(d0, d1)
        val enables = (input_keypoints map {keypoint => keypoint.toUnchecked[T] < value}) ++ Seq(value >= cumsum.last.toUnchecked[T])
        val middle_values = ((input_keypoints zip input_keypoints.tail) zip (cumsum zip cumsum.tail)) map {
          case ((start, end), (left_val, right_val)) =>
            val reciprocal_length = 1 / (end - start)
            val offset = value - start.toUnchecked[T]
            val scaled_amount = offset * reciprocal_length.toUnchecked[T]
            left_val.toUnchecked[T] + scaled_amount * (right_val - left_val).toUnchecked[T]
        }
        val values = Seq(cumsum.head.toUnchecked[T]) ++ middle_values ++ Seq(cumsum.last.toUnchecked[T])
        assert(enables.length == values.length, "Enables should be of the same length as the values")
        priorityMux(enables, values)
      }

      lazy val shape: Seq[I32] = arg.shape
    }
  }

  def PWLCalibration[T:Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg:Readable2D[T])(implicit state: argon.State) = {
    (PWL_mode match {
      case true => PWLCalibration_mux[T] _
      case false => PWLCalibration_Sequential[T] _
    })(pwl_calibration_kernel, input_keypoints)(arg)
  }

    protected def ComputeStrides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: scala.Array[Int] = scala.Array.fill(dimensions.length){1}
    scala.Range(1, dimensions.length, 1) foreach {
      d => {
        strides(d) = strides(d-1) * dimensions(d-1)
      }
    }
    strides
  }

  def Lattice[T : Num](lattice_kernel: scala.Array[scala.Array[scala.Double]], tp: String, shape: scala.Array[scala.Int], units: Int)(arg:Readable2D[T])(implicit state:argon.State, ctx: SrcCtx): Readable2D[T] = {
    import lattice.HypercubeLattice

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.length
    val strides = ComputeStrides(shape)

    // Ignore tp for now, always "hypercube"

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T])}.toSeq
    val params = LUT[OutputType](param_list.length)(param_list:_*)

    // needed to pass shape into readable def
    val lattice_shape = shape

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        // d1 is ignored because it's really only 1-D
        val residualPairs = Seq.tabulate(dimensions) { i =>
          val x = arg(i, d0).to[ResidualType]
          Seq(x.to[AccumResidualType], 1.toFloat.to[AccumResidualType] - x.to[AccumResidualType])
        }
        // Compute all hypervolumes in binary counting order (000, 001, 010, 011, etc.)
        val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(residualPairs: _*)(_ * _)
        // Compute hypercube origin
        // if the dimension is 0, then we optimize by setting the base index to 0 instead.
        val base: Seq[ParameterIndex] = scala.Array.tabulate(dimensions) { x =>
          (lattice_shape(x), PO2Opt) match {
            case (2, true) =>
              0.to[ParameterIndex]
            case _ =>
              arg (x, d0).to[ParameterIndex]
          }
        }
        println(base.mkString(", "))
        // Get all vertices of hypercube and reverse so that these are opposite the hypervolumes
        val corners: Seq[Seq[scala.Int]] = HypercubeLattice.allCorners(Seq.fill(dimensions)(1)).reverse

        // Get flat index for each (corner + origin)
        val indices: Seq[ParameterIndex] = corners map { c =>
          val corner = (base zip c.map(_.to[ParameterIndex])) map { case (a, b) => a + b }
          (corner zip strides) map { case (cc, stride) =>
            cc * stride
          } reduce {
            _ + _
          }
        }

        // Get weighted sum
        hypervolumes.map(_.to[OutputType]).zip(indices).map {
          case (hv, i) =>

          hv * params(i)
        }.reduceTree {
          _ + _
        }
      }

      lazy val shape: Seq[I32] = Seq(arg.shape.head, units)
    }
  }

  def Linear[T:Num](linear_layer_bias: Double, linear_layer_kernel: Array[Array[Double]])(arg: Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        arg(d0, d1)
      }
      lazy val shape: Seq[I32] = Seq(arg.shape.head, 1)
    }
  }
}

object tf {
  def Concatenate[T:Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State): ReadableND[T] = {
    val arg_dims = args.map {_.shape.length}.distinct
    assert(arg_dims.size == 1)

    val dims = arg_dims.head
    val concat_axis = if (axis >= 0) axis else dims + axis

    val axis_sizes = args.map {_.shape(concat_axis)}
    val breakpoints = axis_sizes.scanLeft(0.to[I32])(_ + _)

    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): T = {
        // compute breakpoints for each respective bank. This can be phrased as a priority mux across all banks, with
        // the enable signal being whether the ub[axis] > index[axis]
        val target_index = index(concat_axis)

        val enables: Seq[Bit] = breakpoints map {target_index < _}

        val reads = args.zipWithIndex map {
          case (input: ReadableND[T], input_index) =>
            val sub_index: Seq[I32] = index.zipWithIndex map { case (i, meta_index) =>
              if (meta_index != concat_axis) {
                i
              } else {
                if (input_index == 0) {
                  i
                } else {
                  i - breakpoints(input_index - 1)
                }
              }
            }
            input(sub_index:_*)
        }
        priorityMux(enables, reads)
      }

      lazy val shape: Seq[I32] = {
        // all other dimensions will be a perfect match, except for the concatenated one.
        Range(0, dims, 1) map {
          index =>
            if (index == concat_axis) {
              args.head.shape(index)
            } else {
              (args map {arg: ReadableND[T] => arg.shape(index)}) reduce { _ + _}
            }
        }
      }
    }
  }

  def Minimum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): T = {
        min(arg(index:_*), constant)
      }
      lazy val shape = arg.shape
    }
  }

  def Maximum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): T = {
        max(arg(index:_*), constant)
      }
      lazy val shape = arg.shape
    }
  }
}