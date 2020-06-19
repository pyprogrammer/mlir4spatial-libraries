package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._
import _root_.spatial.dsl

import scala.reflect.ClassTag

object tfl extends PWLCalibration with Lattice {

  private val PWL_mode = sys.env.getOrElse("PWL", "mux") == "mux"

  def CategoricalCalibration[T : Num](categorical_calibration_kernel: scala.Array[scala.Array[scala.Double]])(arg:Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    val units = {
      val lengths = (categorical_calibration_kernel map {_.length}).distinct
      assert(lengths.length == 1, f"Found multiple possible number of units for Categorical Calibration. Expected 1. ${lengths}")
      lengths.head
    }
    val param_list = categorical_calibration_kernel.flatten.map { x => Bits(x.toUnchecked[T]) }.toSeq
    val params = LUT[T](categorical_calibration_kernel.length, units)(param_list:_*)
    new Readable2D[T] {
      override def apply(d0: I32, d1: I32): () => T = {
        val value = arg(d0, d1)()
        val v = params(value.to[I32], d1)
        () => v
      }

      lazy val shape: Seq[I32] = arg.shape
    }
  }

  def Linear[T:Num](linear_layer_bias: Double, linear_layer_kernel: Array[Array[Double]])(arg: Readable2D[T])(implicit state: argon.State): Readable2D[T] = {
    tf.Dense(Array(linear_layer_bias), linear_layer_kernel)(arg)
  }
}

object tf extends Concatenation with Blas3 {

  def Minimum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): () => T = {
        () => min(arg(index:_*)(), constant)
      }
      lazy val shape = arg.shape
    }
  }

  def Maximum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: spatial.dsl.I32*): () => T = {
        () => max(arg(index:_*)(), constant)
      }
      lazy val shape = arg.shape
    }
  }

  def Transpose[T:Num](axes: Array[Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    ShuffleAxes(Seq(axes(0) -> axes(1), axes(1)->axes(0)).toMap)(arg)
  }

  def ShuffleAxes[T:Num](mapping: Map[Int, Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    // Map: Input axis => Original Axis (i.e. 3 -> 2 means accesses along axis 3 are redirected to axis 2
    val dims = arg.shape.length
    mapping flatMap { case (in, out) => Seq(in, out)} foreach {
      axis => assert(axis < dims, s"Attempted to use axis $axis of $dims-Dimensional structure")}

    assert(mapping.values.size == mapping.values.toSeq.distinct.size, s"Target Axes must be unique! $mapping")

    new ReadableND[T] {
      override def apply(index: dsl.I32*): () => T = {
        val remapped = index.zipWithIndex map {
          case (i, dimension) => (mapping.getOrElse(dimension, dimension), i)
        }
        val new_index = remapped sortWith {case (a, b) => a._1 < b._1 } map {_._2}
        arg(new_index:_*)
      }

      override lazy val shape: Seq[dsl.I32] = {
        val remapped = arg.shape.zipWithIndex map {
          case (i, dimension) => (mapping.getOrElse(dimension, dimension), i)
        }
        remapped sortWith {case (a, b) => a._1 < b._1 } map {_._2}
      }
    }
  }

  def GatherV2[T:Num](axis: Int, indices: Array[Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    assert(axis < arg.shape.length, s"Requires axis $axis < ${arg.shape.length}")
    assert(indices.nonEmpty, s"Passed an empty index into Gather")
    val wrapped_axis = if (axis < 0) arg.shape.length - axis else axis

    val lut = LUT[I32](indices.length)((indices map {I32(_)}):_*)

    new ReadableND[T] {
      override def apply(index: dsl.I32*): () => T = {
        val new_index = index.zipWithIndex map {
          case (ind, dim) =>
            if (dim == wrapped_axis) {
              lut(ind)
            } else {
              ind
            }
        }
        arg(new_index:_*)
      }

      override lazy val shape: Seq[dsl.I32] = {
        arg.shape.zipWithIndex map {
          case (s, dim) =>
            if (dim == wrapped_axis) {
              I32(indices.length)
            } else {
              s
            }
        }
      }
    }
  }
}