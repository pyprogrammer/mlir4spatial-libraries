package mlir_libraries

import forge.tags.api
import spatial.libdsl._
import spatial.metadata.memory._

class CoprocessorScope(val coprocessorScopeId: scala.Int, val setupScopeId: scala.Int)(implicit s: argon.State) {
  type T = () => Any
  val state: argon.State = s

  private val coprocessors = scala.collection.mutable.Buffer[T]()

  def register(coproc: T)(implicit srcCtx: SrcCtx) = {
    coprocessors.append(coproc)
  }

  def instantiate(): Void = {
    Stream {
      coprocessors.reverse foreach {x => x()}
    }
  }

  private def escapeToScope[T](id: Int, thunk: => T): T = {
    val bundle = state.bundleStack(id)
    val (result, newBundle) = state.WithScope({
      val tmp = thunk
      tmp
    }, bundle)
    // store the new bundle back
    state.bundleStack.update(coprocessorScopeId, newBundle)
    result
  }

  def escape[T](thunk: => T): T = escapeToScope(coprocessorScopeId, thunk)

//  def setup[T](thunk: => T): T = escapeToScope(setupScopeId, thunk)
}

object CoprocessorScope {
  def apply[T](init: CoprocessorScope => T)(func: (Reg[Bit], T) => Any)(implicit state: argon.State): Void = {
//    println(s"BS Size: ${state.bundleStack.size}")
    val kill: Reg[Bit] = Reg[Bit](false, "CoprocessorScopeKill")
    val setupScopeId: Int = state.bundleStack.size
    val coprocScopeId: Int = state.bundleStack.size
    println(s"BS Size: ${state.bundleStack.size}")
    if (Options.Coproc) {
      'CoprocessorScope.Stream(breakWhen = kill).Foreach(I32(1) by I32(1)) {
        _ =>
          // Stage into current scope
          val scope = new CoprocessorScope(coprocScopeId, setupScopeId)
          val initialized = init(scope)
          func(kill, initialized)
//            kill := 1.to[Bit]
          scope.instantiate()
      }
    } else {
      'CoprocessorScope.Pipe {
        val scope = new CoprocessorScope(coprocScopeId, setupScopeId)
        val initialized = init(scope)
        func(kill, initialized)
      }
    }
  }
}


// The coprocessor scope defines the control stream.
abstract class Coprocessor[In_T: Bits, Out_T: Bits](input_arity: Int, output_arity: Int) {
  def coprocessorScope: CoprocessorScope
  implicit val state: argon.State = coprocessorScope.state

  coprocessorScope.register(instantiate)

  protected val SCALE_FACTOR = 4
  protected val INPUT_FIFO_DEPTH = 128
  protected val OUTPUT_FIFO_DEPTH = 16

  // Coprocessors have input fifo sets, output fifo sets, and a control stream.
  // Assume for now that all inputs are of the same type, and all outputs are of the same type.
  protected val id_fifos = collection.mutable.Buffer[FIFO[I32]]()
  protected val input_fifos = collection.mutable.Buffer[Seq[FIFO[In_T]]]()
  protected val output_fifos = collection.mutable.Buffer[Seq[FIFO[Out_T]]]()


  var frozen: Boolean = false

//  // Override this for the core inner function.
//  def execute(inputs: Seq[In_T]): Seq[Out_T]

  def enq(inputs: Seq[In_T])

  def deq(inputs: Seq[In_T]): Seq[Out_T]

  def instantiate(): Void = {
    assert(!frozen, "Shouldn't be frozen yet")
    frozen = true

    if (id_fifos.size == 0) {
      argon.warn("Instantiated coprocessor has no uses")
      state.logWarning()
    }

    // The central fifo should be able to handle 1 input per input fifo, plus
    val central_input_fifos = Range(0, input_arity) map { _ => FIFO[In_T](I32(input_fifos.size * SCALE_FACTOR)) }
    central_input_fifos.zipWithIndex foreach {
      case (fifo, index) =>
        fifo.explicitName = f"CentralInputFifo$index"
    }
    val central_output_indices = FIFO[I32](I32(input_fifos.size * SCALE_FACTOR))
    central_output_indices.explicitName = "CentralOutputIndicesFifo"

    val intermediateArbiterFifo = FIFO[I32](I32(INPUT_FIFO_DEPTH))
    'CoprocessorPreArbiter.Stream.Foreach(*) {
      _ =>
        val ind = priorityDeq(id_fifos:_*)
        intermediateArbiterFifo.enq(ind)
    }

    // now execute the actual kernel
    'CoprocessorArbiter.Stream.Foreach(*) {
      _ => {
        // dequeues from all of the fifoes which have an element
        println(s"Number of id fifos: ${id_fifos.size}")
//        val next_fifo = priorityDeq(id_fifos: _*)
        val next_fifo = intermediateArbiterFifo.deq
        val should_dequeue = input_fifos.zipWithIndex map {
          case (_, ind) =>
            I32(ind) === next_fifo
        }

        Pipe { // Make it not buffer this reg
          val desired_fifo_has_data = Reg[Bit](false)
          desired_fifo_has_data := false
          // Wait for desired fifo to have some data.  May spin for a bit in the beginning but eventually it should only spin for a few cycles per request
          Sequential(breakWhen = desired_fifo_has_data).Foreach(*) { _ =>
            val keep_spinning = should_dequeue zip input_fifos map {
              case (sd, fifo_list) =>
                val anyEmpty = (fifo_list map {_.isEmpty}).reduceTree {_ || _}
                // If any are empty and sd is high, then we need to keep spinning
                anyEmpty && sd
            } reduceTree { _ || _ }
            desired_fifo_has_data := keep_spinning
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

          val final_values = reduced map {
            case (v, _) =>
              val r = Reg[In_T]
              r := v
              r.value
          }

          enq(final_values)

          (final_values zip central_input_fifos) foreach {
            case (value, fifo) =>
              fifo.enq(value)
          }
          central_output_indices.enq(next_fifo)
        }
      }
    }

    'Coprocessor.Stream.Foreach(*) {
      _ =>
        Pipe {
          val input_registers = central_input_fifos map {
            _ =>
              Reg[In_T]
          }
          input_registers.zipWithIndex foreach {
            case (reg, ind) =>
              reg.explicitName = f"CoprocessorInputRegister$ind"
          }

          input_registers.zip(central_input_fifos) foreach {
            case (reg, fifo) =>
              reg := fifo.deq
          }

          val inputs = input_registers map {_.value}

          val destination = central_output_indices.deq
          val results = deq(inputs)
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
  }

  // Process function takes an input read from a fifo and writes to the corresponding output fifo.

  class CoprocessorInterface(input_stream: Seq[FIFO[In_T]], output_stream: Seq[FIFO[Out_T]], identity_fifo: FIFO[I32], id: Int) {
    var enqueued = false
    var dequeued = false

    def enq(input: Seq[In_T], en: Bit = Bit(true)): Void = {
      enqueued = true
      Sequential {
        Parallel {
          (input_stream zip input) foreach {
            case (fifo, in) => fifo.enq(in, en)
          }
        }
        identity_fifo.enq(I32(id), en)
      }
    }

    def deq(en: Bit = Bit(true)): Seq[Out_T] = {
      dequeued = true
      output_stream map {
        _.deq(en)
      }
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
    println(s"CoprocBundle: $io, $id")
    new CoprocessorInterface(io._1, io._2, io._3, id)
  }
}
