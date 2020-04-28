package tensorflow_lattice

import mlir_libraries.types._

import spatial.libdsl._

object tfl {

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

  def PWLCalibration[T:Num](pwl_calibration_kernel: scala.Array[scala.Array[scala.Double]], input_keypoints: scala.Array[scala.Double])(arg:Readable2D[T])(implicit state: argon.State) = {
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

  protected def ComputeStrides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: scala.Array[Int] = scala.Array.fill(dimensions.length){1}
    scala.Range(1, dimensions.length, 1) foreach {
      d => {
        strides(d) = strides(d-1) * dimensions(d-1)
      }
    }
    strides
  }

  def Lattice[T : Num](lattice_kernel: scala.Array[scala.Array[scala.Double]], tp: String, shape: scala.Array[Int])(arg:Readable2D[T])(implicit state:argon.State, ctx: SrcCtx): Readable2D[T] = {
    import lattice.HypercubeLattice

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.length
    val strides = ComputeStrides(shape)

    // Ignore tp for now, always "hypercube"

    val param_list = lattice_kernel.flatten.map { x => Bits(x.to[T])}.toSeq
    val params = LUT[OutputType](param_list.length)(param_list:_*)

    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): T = {
        val residualPairs = Seq.tabulate(dimensions) { i =>
          val x = arg(i, d0).to[ResidualType]
          Seq(x.to[AccumResidualType], 1.toFloat.to[AccumResidualType] - x.to[AccumResidualType])
        }
        // Compute all hypervolumes in binary counting order (000, 001, 010, 011, etc.)
        val hypervolumes: Seq[AccumResidualType] = HypercubeLattice.CombinationTree(residualPairs: _*)(_ * _)
        // Compute hypercube origin
        val base: Seq[ParameterIndex] = scala.Array.tabulate(dimensions) { x => arg(x, d0).to[ParameterIndex] }
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
        hypervolumes.map(_.to[OutputType]).zip(indices).map { case (hv, i) => hv * params(i) }.reduceTree {
          _ + _
        }
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
}