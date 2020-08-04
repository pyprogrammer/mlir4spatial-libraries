package mlir_libraries

import forge.tags.api
import spatial.libdsl._
import spatial.metadata.memory._

class CoprocessorScope(val scope_id: scala.Int)(implicit s: argon.State) {
  type T = () => Any
  val state: argon.State = s

  private val coprocessors = scala.collection.mutable.Buffer[T]()

  def register(coproc: T)(implicit srcCtx: SrcCtx) = {
    coprocessors.append(coproc)
  }

  def instantiate(): Void = {
    Stream {
      coprocessors foreach {x => x()}
    }
  }

  def escape[T](thunk: => T): T = {
    val bundle = state.bundleStack(scope_id)
    val (result, newBundle)  = state.WithScope({
      val tmp = thunk
      tmp
    }, bundle)
    // store the new bundle back
    state.bundleStack.update(scope_id, newBundle)
    result
  }
}

object CoprocessorScope {
  @api def apply[T](init: CoprocessorScope => T)(func: T => Any): Void = {
    val kill: Reg[Bit] = Reg[Bit](false, "CoprocessorScopeKill")
    val scope_id: Int = state.bundleStack.size - 1
    Stream(breakWhen = kill).Foreach(*) {
      _ =>
        val scope = new CoprocessorScope(scope_id)
        val initialized = init(scope)
        Pipe {
          func(initialized)
          kill := 1.to[Bit]
        }
        scope.instantiate()
    }
  }
}


// The coprocessor scope defines the control stream.
// Todo(pyprogrammer): prealloc should be removed once we figure out how to stash the argon.State
//   We should be able to directly create new FIFOs in the parent scope.
abstract class Coprocessor[In_T: Bits, Out_T: Bits](input_arity: Int, output_arity: Int) {
  def coprocessorScope: CoprocessorScope
  implicit val state: argon.State = coprocessorScope.state

  coprocessorScope.register(instantiate)

  protected val SCALE_FACTOR = 4
  protected val INPUT_FIFO_DEPTH = 16
  protected val OUTPUT_FIFO_DEPTH = 16

  // Coprocessors have input fifo sets, output fifo sets, and a control stream.
  // Assume for now that all inputs are of the same type, and all outputs are of the same type.
  protected val id_fifos = collection.mutable.Buffer[FIFO[I32]]()
  protected val input_fifos = collection.mutable.Buffer[Seq[FIFO[In_T]]]()
  protected val output_fifos = collection.mutable.Buffer[Seq[FIFO[Out_T]]]()


  var frozen: Boolean = false

  // Override this for the core inner function.
  def execute(inputs: Seq[In_T]): Seq[Out_T]


  def instantiate(): Void = {
    assert(!frozen)
    frozen = true

    // The central fifo should be able to handle 1 input per input fifo, plus
    val central_input_fifos = Range(0, input_arity) map { _ => FIFO[In_T](I32(input_fifos.size * SCALE_FACTOR)) }
    val central_output_indices = FIFO[I32](I32(input_fifos.size * SCALE_FACTOR))

    // now execute the actual kernel
    'CoprocessorArbiter.Stream.Foreach(*) {
      _ => {

        // dequeues from all of the fifoes which have an element
        val next_fifo: I32 = priorityDeq(id_fifos:_*)

        val debug: Reg[I32] = Reg[I32](I32(-1), "NEXT_FIFO").dontTouch

        debug := next_fifo

        val should_dequeue = input_fifos.zipWithIndex map {
          case (_, ind) =>
            I32(ind) === next_fifo
        }

        val dequeued = (input_fifos zip should_dequeue) map {
          case (bundle, en) =>
            bundle map {
              fifo => (fifo.deq(en), en)
            }
        }

        val reduced = dequeued.transpose map {
          signals =>
            signals reduceTree {
              case ((v1, en1), (v2, en2)) =>
                (mux(en1, v1, v2), en1 || en2)
            }
        }

        (reduced zip central_input_fifos) foreach {
          case ((value, _), fifo) =>
            fifo.enq(value)
        }
        central_output_indices.enq(next_fifo)
        }
    }

    'Coprocessor.Stream.Foreach(*) {
      _ =>
        val inputs = central_input_fifos map {
          _.deq
        }
        val destination = central_output_indices.deq
        val results = execute(inputs)

        // writeback to proper fifo.
        output_fifos.zipWithIndex foreach {
          case (output_bundle, output_index) =>
            val write_enable = I32(output_index) === destination
            (output_bundle zip results) foreach {
              case (fifo, result) =>
                fifo.enq(result, write_enable)
            }
        }
    }
  }

  // Process function takes an input read from a fifo and writes to the corresponding output fifo.

  class CoprocessorInterface(input_stream: Seq[FIFO[In_T]], output_stream: Seq[FIFO[Out_T]], identity_fifo: FIFO[I32], id: I32) {
    def enq(input: Seq[In_T]): Void = {
      identity_fifo.enq(id)

      (input_stream zip input) foreach {
        case (fifo, in) => fifo.enq(in)
      }
    }

    def deq(): Seq[Out_T] = {
      output_stream map {_.deq}
    }
  }

  var cnt = 0
  def interface: CoprocessorInterface = {
    val id = cnt
    cnt += 1

    val io = coprocessorScope.escape {
      val new_input_fifo_set = Range(0, input_arity) map {
        i =>
          val f = FIFO[In_T](I32(INPUT_FIFO_DEPTH))
          f.explicitName = f"InputFifo${id}_$i"
          f
      }
      input_fifos.append(new_input_fifo_set)

      val id_fifo = FIFO[I32](I32(INPUT_FIFO_DEPTH))
      id_fifo.explicitName = f"IdentityFifo$id"
      id_fifos.append(id_fifo)

      val output_fifo_set = Range(0, output_arity) map {
        i =>
          val f = FIFO[Out_T](I32(OUTPUT_FIFO_DEPTH))
          f.explicitName = f"OutputFifo${id}_$i"
          f
      }
      output_fifos.append(output_fifo_set)

      (new_input_fifo_set, output_fifo_set, id_fifo)
    }
    new CoprocessorInterface(io._1, io._2, io._3, I32(id))
  }
}
