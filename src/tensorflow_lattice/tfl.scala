package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._
import _root_.spatial.dsl

import scala.reflect.ClassTag

object tfl extends PWLCalibration {

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

  def Lattice[T : Num](lattice_kernel: scala.Array[scala.Array[scala.Double]], tp: String, shape: scala.Array[scala.Int], units: Int)(arg:ReadableND[T])(implicit state:argon.State, ctx: SrcCtx): Readable2D[T] = {
    import lattice.HypercubeLattice

    type ResidualType = T
    type AccumResidualType = T
    type ParameterIndex = I32
    type OutputType = T

    val dimensions = shape.length
    val strides = ComputeStrides(shape)

    // Ignore tp for now, always "hypercube"

    // The outer list is in order, the inner dimension of the array corresponds to an output vector unit.

    val param_list = lattice_kernel.flatten.map { x => Bits(x.toUnchecked[T])}.toSeq
    val params = LUT[OutputType](lattice_kernel.length, units)(param_list:_*)

    // needed to pass shape into readable def
    val lattice_shape = shape

    val expanded_arg = if (arg.shape.length == 3) {
      arg
    } else {
      tf.expand_dims(axis=2)(arg)
    }
    assert(expanded_arg.shape.length == 3)


    new Readable2D[T] {
      override def apply(batch: I32, unit: I32): T = {
        val residualPairs = Seq.tabulate(dimensions) { i =>
          val x = expanded_arg(batch, i, unit).to[ResidualType]
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
              expanded_arg(batch, x, unit).to[ParameterIndex]
          }
        }
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

          hv * params(i, unit)
        }.reduceTree {
          _ + _
        }
      }

      // A lattice goes from (batch, dim, unit) -> (batch, unit)
      lazy val shape: Seq[I32] = Seq(arg.shape.head, units)
    }
  }

  def Linear[T:Num:ClassTag](linear_layer_bias: Double, linear_layer_kernel: Array[Array[Double]])(arg: Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    val input_units = linear_layer_kernel.length
    val output_units = {
      val output_shapes = (linear_layer_kernel map {_.length}).distinct
      assert(output_shapes.length == 1, s"Found multiple possible output shapes: $output_shapes")
      output_shapes.head
    }

    val kernel_lut = LUT[T](input_units, output_units)((linear_layer_kernel.flatten map {x => Bits(x.toUnchecked[T])}):_*)

    new Readable2D[T] {
      override def apply(batch: I32, dim: I32): T = {
        // Goes from (batch x input units) x kernel^T (input x output) -> batch x output units.
        // Given C = AB, we have C_ij = Sum_k A_ik B_kj
        val result = Reg[T](linear_layer_bias.toUnchecked[T])
//        Fold(result)(I32(input_units) by I32(1) par I32(input_units)) {
//          inner: I32 =>
//            arg(batch, inner) * kernel_lut(inner, dim)
//        } {_ + _}
        linear_layer_bias.toUnchecked[T] + (Array.tabulate(input_units) { i => arg(batch, i) * kernel_lut(I32(i), dim)}).reduce{_+_}
      }
      lazy val shape: Seq[I32] = Seq(arg.shape.head, I32(output_units))
    }
  }
}

object tf extends Concatenation {

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