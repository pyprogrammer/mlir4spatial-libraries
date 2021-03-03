package tensorflow_lattice
import mlir_libraries.CoprocessorScope
import mlir_libraries.types._
import spatial.dsl
import spatial.libdsl._

trait Concatenation {
  def Concatenate[T:Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State, cps: CoprocessorScope): ReadableND[T] = {
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

    implicit val vecBitsEV: Bits[Vec[I32]] = Vec.fromSeq(Range(0, dims) map {_ => I32(0)})
    @struct case class IndexEnablePair(index: Vec[I32], enable: Bit)

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
          def computeEnables(index: Seq[dsl.I32], ens: Set[dsl.Bit]) = {
            // compute breakpoints for each respective bank. This can be phrased as a priority mux across all banks, with
            // the enable signal being whether the ub[axis] > index[axis] && index[axis] >= ub[axis-1]

            val target_index = index(concat_axis)

            val enables: Seq[Bit] = breakpoints.zipWithIndex map {
              case (bp, ind) =>
                if (ind == 0) {
                  target_index < bp
                } else {
                  breakpoints(ind - 1) <= target_index && target_index < bp
                }
            }
            enables
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

          val fwdEnableFIFOs = interfaces map {_ =>
            val fwdEnableFIFO = FIFO[IndexEnablePair](I32(128))
            fwdEnableFIFO
          }

          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {
            assert(index.size == dims, "Index and dims should match up!")
            val enables = computeEnables(index, ens)
            enables.zipWithIndex foreach {
              case (en, input_index) =>
                val sub_index = get_new_index(index, input_index)
                argon.stage(spatial.node.FIFOEnq(fwdEnableFIFOs(input_index), IndexEnablePair(Vec.fromSeq(sub_index), en), ens))
                interfaces(input_index).enq(sub_index, ens + en)
            }
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            val meta = fwdEnableFIFOs map { x => argon.stage(spatial.node.FIFODeq(x, ens))}
            val enables = meta map {_.enable}
            val reads = (meta zip interfaces) map {
              case (met, interface) =>
                val ind = met.index
                val new_index = Range(0, dims) map {x => ind(x)}
                interface.deq(new_index, ens + met.enable)
            }

            val v = oneHotMux(enables, reads)

            mlir_libraries.debug_utils.TagVector("ConcatenateOutput", Seq(v), ens)
            mlir_libraries.debug_utils.TagVector("ConcatenateIndex", index, ens)

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

  def Stack[T: Num](axis: Int)(args:ReadableND[T]*)(implicit state:argon.State, cps: CoprocessorScope): ReadableND[T] = {
    // tf.Stack inserts a new axis at the specified location.
    // We achieve this by shimming a new ReadableND in, and then Concatenating them.
    val expanded = args map { x => expand_dims(axis)(x)}
    Concatenate(axis)(expanded:_*)
  }
}
