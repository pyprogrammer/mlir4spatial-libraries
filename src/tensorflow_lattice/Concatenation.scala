package tensorflow_lattice

import mlir_libraries.types._
import spatial.dsl
import spatial.libdsl._

trait Concatenation {
  def Concatenate[T:Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State): ReadableND[T] = {
    val arg_dims = args.map {
      _.shape.length
    }.distinct
    assert(arg_dims.size == 1)

    val dims = arg_dims.head
    val concat_axis = if (axis >= 0) axis else dims + axis

    val axis_sizes = args.map {
      _.shape(concat_axis)
    }
    val breakpoints = axis_sizes.tail.scanLeft(axis_sizes.head) {
      _ + _
    }
    println(f"Axis Sizes: $axis_sizes")
    println(f"Concatenate Breakpoints: $breakpoints")

    new ReadableND[T] {
      lazy val shape: Seq[I32] = {
        // all other dimensions will be a perfect match, except for the concatenated one.
        Range(0, dims, 1) map {
          index =>
            if (index != concat_axis) {
              args.head.shape(index)
            } else {
              (args map { arg: ReadableND[T] => arg.shape(index) }) reduceTree {
                _ + _
              }
            }
        }
      }

      override def getInterface: Interface[T] = {
        val interfaces = args map {_.getInterface}
        new Interface[T] {
          def prune_args(index: Seq[dsl.I32], ens: Set[dsl.Bit]) = {
            // compute breakpoints for each respective bank. This can be phrased as a priority mux across all banks, with
            // the enable signal being whether the ub[axis] > index[axis] && index[axis] >= ub[axis-1]
            val target_index = Reg[I32]
            target_index := index(concat_axis)

            val enables: Seq[Bit] = breakpoints.zipWithIndex map {
              case (bp, ind) =>
                if (ind == 0) {
                  target_index < bp
                } else {
                  breakpoints(ind - 1) <= target_index && target_index < bp
                }
            }

            // Quick optimization in the case where enables has a set of false values followed by either a true or an
            // undetermine value:
            // prune all enables for which it is statically false.
            // after this, prune all elements from the rear after the first statically true one.

            val false_pruned_enables = enables.zipWithIndex filter {
              case (en, ind) =>
                en match {
                  case argon.Const(x) =>
                    // If it's a constant, we only keep it if it's true.
                    x.toBoolean
                  case _ =>
                    // Otherwise, we don't know what it is and have to keep it.
                    true
                }
            }

            val first_true = false_pruned_enables find {
              case (en, ind) => en.isConst
            }

            val pruned_enables = false_pruned_enables filter {
              // If first_true, then we drop everything after. Otherwise, keep everything.
              case (en, ind) =>
                first_true match {
                  case Some((_, index)) =>
                    ind <= index
                  case None =>
                    true
                }
            }

            assert(pruned_enables.nonEmpty, "Cannot have an empty enable list.")
            pruned_enables
          }

          def get_new_index(index: Seq[dsl.I32], input_index: Int) = {
            index.zipWithIndex map { case (i, meta_index) =>
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
          }

          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {

            val pruned_enables = prune_args(index, ens)
            pruned_enables foreach {
              case (en, input_index) =>
                val sub_index = get_new_index(index, input_index)
                interfaces(input_index).enq(sub_index, ens + en)
            }
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            mlir_libraries.debug_utils.TagVector("ConcatenationInput", index, ens)
            val pruned_enables = prune_args(index, ens)
            mlir_libraries.debug_utils.TagVector("ConcatenationEnables", pruned_enables map {_._1}, ens)
            val reads = (pruned_enables zip interfaces) map {
              case ((en, ind), interface) =>
                val sub_index = get_new_index(index, ind)
                mlir_libraries.debug_utils.TagVector("ConcatenateSubIndex", sub_index, ens + en)
                interface.deq(sub_index, ens + en)
            }
            val v = priorityMux(pruned_enables map {
              _._1
            }, reads)

            mlir_libraries.debug_utils.TagVector("ConcatenateOutput", Seq(v), ens)

            v
          }
        }
      }
    }
  }

  def expand_dims[T: Num](axis: Int)(arg: ReadableND[T])(implicit state:argon.State): ReadableND[T] = {

    new ReadableND[T] {
      override def getInterface: Interface[T] = {
        val subInterface = arg.getInterface
        new Interface[T] {
          def getIndex(index:Seq[dsl.I32]) = {
            index.take(axis) ++ index.drop(axis + 1)
          }
          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {
            subInterface.enq(getIndex(index), ens)
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            subInterface.deq(getIndex(index), ens)
          }
        }
      }
      override lazy val shape: Seq[dsl.I32] = {
        arg.shape.take(axis) ++ Seq(I32(1)) ++ arg.shape.drop(axis)
      }
    }
  }

  def Stack[T: Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State): ReadableND[T] = {
    // tf.Stack inserts a new axis at the specified location.
    // We achieve this by shimming a new ReadableND in, and then Concatenating them.
    val expanded = args map { x => expand_dims(axis)(x)}
    Concatenate(axis)(expanded:_*)
  }
}
