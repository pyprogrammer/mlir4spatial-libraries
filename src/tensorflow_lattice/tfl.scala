package tensorflow_lattice

import mlir_libraries.types._
import spatial.libdsl._
import _root_.spatial.dsl

import scala.reflect.ClassTag
import mlir_libraries.{Tensor => MLTensor}

object tfl extends PWLCalibration with Lattice {

  def CategoricalCalibration[T : Num](categorical_calibration_kernel: MLTensor[scala.Double])(arg:ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    val categorical_calibration_kernel_array = categorical_calibration_kernel.to2DSeq
    val units = {
      val lengths = (categorical_calibration_kernel_array map {_.length}).distinct
      assert(lengths.length == 1, f"Found multiple possible number of units for Categorical Calibration. Expected 1. ${lengths}")
      lengths.head
    }

    val degenerate_PWL_input = arg.shape(1) match {
      case argon.Const(c) => c.value == 1
      case _ => false
    }

    val param_list = categorical_calibration_kernel_array.flatten.map { x => Bits(x.toUnchecked[T]) }.toSeq
    val params = LUT[T](categorical_calibration_kernel_array.length, units)(param_list:_*)
    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val initial_ind = index.dropRight(1)
        val unit = index.last
        val value = if (degenerate_PWL_input) { arg(initial_ind :+ I32(0), ens)() } else { arg(index, ens)() }
        val v = params(value.to[I32], unit)
        () => v
      }

      lazy val shape: Seq[I32] = arg.shape
    }
  }

  def Linear[T:Num](linear_layer_bias: Double, linear_layer_kernel: MLTensor[Double])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    val bias_size = linear_layer_kernel.shape(1)
    val bias_values = Array.fill(bias_size)(linear_layer_bias)
    tf.Dense(MLTensor(values=bias_values, shape=Array(bias_size)), linear_layer_kernel)(arg)
  }
}

object tf extends Concatenation with Blas3 {

  def Minimum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val tmp = arg(index, ens)
        () => min(tmp(), constant)
      }
      lazy val shape = arg.shape
    }
  }

  def Maximum[T:Num](constant: Double)(arg:ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val tmp = arg(index, ens)
        () => max(tmp(), constant)
      }
      lazy val shape = arg.shape
    }
  }

  def Transpose[T:Num](axes: MLTensor[Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    assert(axes.rank == 1)
    ShuffleAxes(Seq(axes(0) -> axes(1), axes(1)->axes(0)).toMap)(arg)
  }

  def ShuffleAxes[T:Num](mapping: Map[Int, Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    // Map: Input axis => Original Axis (i.e. 3 -> 2 means accesses along axis 3 are redirected to axis 2
    val dims = arg.shape.length
    mapping flatMap { case (in, out) => Seq(in, out)} foreach {
      axis => assert(axis < dims, s"Attempted to use axis $axis of $dims-Dimensional structure")}

    assert(mapping.values.size == mapping.values.toSeq.distinct.size, s"Target Axes must be unique! $mapping")

    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        val remapped = index.zipWithIndex map {
          case (i, dimension) => (mapping.getOrElse(dimension, dimension), i)
        }
        val new_index = remapped sortWith {case (a, b) => a._1 < b._1 } map {_._2}
        arg(new_index, ens)
      }

      override lazy val shape: Seq[dsl.I32] = {
        val remapped = arg.shape.zipWithIndex map {
          case (i, dimension) => (mapping.getOrElse(dimension, dimension), i)
        }
        remapped sortWith {case (a, b) => a._1 < b._1 } map {_._2}
      }
    }
  }

  def GatherV2[T:Num](axis: Int, indices: MLTensor[Int])(arg: ReadableND[T])(implicit state: argon.State): ReadableND[T] = {
    assert(axis < arg.shape.length, s"Requires axis $axis < ${arg.shape.length}")
    val wrapped_axis = if (axis < 0) arg.shape.length - axis else axis

    val flattened_indices = indices.flatten
    val lut = LUT[I32](flattened_indices.length)((flattened_indices map {I32(_)}):_*)

    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        // The first and last parts of the index are untouched.
        val initial_index = index.take(wrapped_axis)
        val last_index = index.drop(wrapped_axis + indices.rank)

        // Remap the middle part.
        val mid_index: Seq[I32] = index.slice(wrapped_axis, wrapped_axis + indices.rank)
        assert(mid_index.length == indices.rank, "Index Rank is not consistent")

        // The mid index yields an index into the LUT which corresponds to the original index.
        val strides = indices.strides map {I32(_)}
        val flattened_index: I32 = ((mid_index zip strides) map {
          case (a, b) => a * b
        }) reduceTree {_ + _}

        val new_index = initial_index ++ Seq(lut(flattened_index)) ++ last_index

        {
          import spatial.dsl._
          print(r"Input Index:")
          index foreach {x => print(r" ${x}")}
          println(r"")
          print(r"Output Index:")
          new_index foreach {x => print(r" ${x}")}
          println(r"")
        }

        arg(new_index, ens)
      }

      override lazy val shape: Seq[dsl.I32] = {
        arg.shape.take(wrapped_axis) ++ indices.shape.map {I32(_)} ++ arg.shape.drop(wrapped_axis + 1)
      }
    }
  }
}