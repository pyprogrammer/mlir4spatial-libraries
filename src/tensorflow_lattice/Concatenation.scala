package tensorflow_lattice

import mlir_libraries.types._
import spatial.dsl
import spatial.libdsl._

trait Concatenation {
  def Concatenate[T:Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State): ReadableND[T] = {
    val arg_dims = args.map {_.shape.length}.distinct
    assert(arg_dims.size == 1)

    val dims = arg_dims.head
    val concat_axis = if (axis >= 0) axis else dims + axis

    val axis_sizes = args.map {_.shape(concat_axis)}
    val breakpoints = axis_sizes.tail.scanLeft(axis_sizes.head){_ + _}
    println(f"Concatenate Breakpoints: $breakpoints")

    new ReadableND[T] {
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {

        // compute breakpoints for each respective bank. This can be phrased as a priority mux across all banks, with
        // the enable signal being whether the ub[axis] > index[axis]
        val target_index = index(concat_axis)

        val enables: Seq[Bit] = breakpoints map {target_index < _}

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
        if (pruned_enables.length > 1) {
          println(s"Start: Concatenate Read Index: $index")
          println(s"Pruned Enables: $pruned_enables")
        }

        val reads = pruned_enables map {
          case (en, input_index) =>
            val input = args(input_index)
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
            input(sub_index, ens + en)
        }
        assert(pruned_enables.length == reads.length,
          f"Enables (${enables.length}) and Reads (${reads.length}) should have same length.")

        val final_signals = pruned_enables map {_._1}

        {
          import spatial.dsl._
          print(r"Concatenate Signals: ")
          final_signals foreach {sig => print(r" $sig")}
          println("")
        }

        if (pruned_enables.length == 1) {
          reads.head
        } else {
          () => priorityMux(pruned_enables map {_._1}, reads map {x => x()})
        }
      }

      lazy val shape: Seq[I32] = {
        // all other dimensions will be a perfect match, except for the concatenated one.
        Range(0, dims, 1) map {
          index =>
            if (index != concat_axis) {
              args.head.shape(index)
            } else {
              (args map {arg: ReadableND[T] => arg.shape(index)}) reduceTree {_ + _}
            }
        }
      }
    }
  }

  def expand_dims[T: Num](axis: Int)(arg: ReadableND[T])(implicit state:argon.State): ReadableND[T] = {
    new ReadableND[T] {
      override lazy val shape: Seq[dsl.I32] = {
        arg.shape.take(axis) ++ Seq(I32(1)) ++ arg.shape.drop(axis)
      }

      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        // cut out middle dimension.
        val new_index = index.take(axis) ++ index.drop(axis + 1)
        arg(new_index, ens)
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
