package mlir_libraries

import forge.tags.api
import spatial.libdsl._
import spatial.metadata.memory._

class CoprocessorScope(implicit s: argon.State) {
  type Command = Bit

  type T = FIFO[Command] => Any
  val state: argon.State = s

  private val command_fifos = scala.collection.mutable.Buffer[FIFO[Command]]()
  private val coprocessors = scala.collection.mutable.Buffer[T]()

  def register(coproc: T)(implicit srcCtx: SrcCtx) = {
    coprocessors.append(coproc)
    val command_fifo = FIFO[Command](I32(1))
    command_fifo.explicitName = s"CommandFifo_${command_fifos.size}"
    command_fifos.append(command_fifo)

    command_fifo
  }

  def kill(enable: Option[Bit] = None)(implicit srcCtx: SrcCtx): Void = {
    utils.checkpoint("PreKill")
    command_fifos.foreach {
      fifo =>
        enable match {
          case Some(en) => fifo.enq(1.to[Bit], en)
          case None => fifo.enq(1.to[Bit])
        }
    }
    utils.checkpoint("PostKill")
  }

  def instantiate(): Void = {
    Stream {
      (coprocessors zip command_fifos) foreach {
        case (co, fifo) =>
          co(fifo)
      }
    }
  }
}

object CoprocessorScope {
  @api def apply[T](init: CoprocessorScope => T)(func: T => Any)(implicit state: argon.State): Void = {
    Stream {
      val scope = new CoprocessorScope()
      val initialized = init(scope)
      Pipe {
        func(initialized)
        scope.kill()
      }
      scope.instantiate()
    }
  }
}


// The coprocessor scope defines the control stream.
// Todo(pyprogrammer): prealloc should be removed once we figure out how to stash the argon.State
//   We should be able to directly create new FIFOs in the parent scope.
abstract class Coprocessor[In_T: Bits, Out_T: Bits](input_arity: Int, output_arity: Int, prealloc: Int) {
  def coprocessorScope: CoprocessorScope
  implicit val state: argon.State = coprocessorScope.state

  coprocessorScope.register(instantiate)

  protected val SCALE_FACTOR = 4
  protected val INPUT_FIFO_DEPTH = 2
  protected val OUTPUT_FIFO_DEPTH = 2

  // Coprocessors have input fifo sets, output fifo sets, and a control stream.
  // Assume for now that all inputs are of the same type, and all outputs are of the same type.
  protected val input_fifos = collection.mutable.Buffer[Seq[FIFO[In_T]]]()
  protected val output_fifos = collection.mutable.Buffer[Seq[FIFO[Out_T]]]()


  var frozen: Boolean = false

  // Override this for the core inner function.
  def execute(inputs: Seq[In_T]): Seq[Out_T]

  var cnt = 0
  Range(0, prealloc) foreach {
    pre =>
      val new_input_fifo_set = Range(0, input_arity) map {
        i =>
          val f = FIFO[In_T](I32(INPUT_FIFO_DEPTH))
          f.explicitName = f"InputFifo$pre$i"
          f
      }
      input_fifos.append(new_input_fifo_set)

      val output_fifo_set = Range(0, output_arity) map {
        i =>
          val f = FIFO[Out_T](I32(OUTPUT_FIFO_DEPTH))
          f.explicitName = f"OutputFifo$pre$i"
          f
      }
      output_fifos.append(output_fifo_set)
  }


  def instantiate(command_queue: FIFO[CoprocessorScope#Command]): Void = {
    assert(!frozen)
    frozen = true

    // The central fifo should be able to handle 1 input per input fifo, plus
    val central_input_fifos = Range(0, input_arity) map { _ => FIFO[In_T](I32(input_fifos.size * SCALE_FACTOR)) }
    val central_output_indices = FIFO[I32](I32(input_fifos.size * SCALE_FACTOR))

    val arbiter_command = FIFO[CoprocessorScope#Command](I32(1))
    arbiter_command.explicitName = "ArbiterCommandFIFO"

    val processor_command = FIFO[CoprocessorScope#Command](I32(1))
    processor_command.explicitName = "ProcessorCommandFIFO"

    val main_break: Reg[Bit] = Reg[Bit](false, "coprocessor_main")
    'CoprocessorController.Stream(breakWhen = main_break).Foreach(*) {
      _ => {
        Pipe {
          utils.checkpoint("CoprocControllerActive")
          val en = !command_queue.isEmpty
          val result = command_queue.deq(en)
          Parallel {
            arbiter_command.enq(result, en)
            processor_command.enq(result, en)
          }
          main_break.write(result, en)
        }
      }
    }

    // now execute the actual kernel
    val arbiter_break: Reg[Bit] = Reg[Bit](false, "arbiter_break")
    'CoprocessorArbiter.Stream(breakWhen = arbiter_break).Foreach(*) {
      _ => {
        utils.checkpoint("CoprocArbiterActive")
        // dequeues from all of the fifoes which have an element
        val has_elements = input_fifos map {
          bundle => bundle map {
            !_.isEmpty
          } reduceTree {
            _ && _
          }
        }

        // check if output is almost full. Don't enqueue if that's the case.
//        val has_backpressure = output_fifos map { bundle =>
//          bundle map {
//            _.isAlmostFull
//          } reduceTree {
//            _ || _
//          }
//        }

//        val should_dequeue = (has_elements zip has_backpressure) map { case (a, b) => a && b }
        val should_dequeue = has_elements

        val dequeued = (input_fifos zip should_dequeue) map {
          case (bundle, en) =>
            bundle map {
              fifo => fifo.deq(en)
            }
        }

        (dequeued zip should_dequeue).zipWithIndex foreach {
          case ((bundle, en), ind) =>
            (bundle zip central_input_fifos) foreach {
              case (value, fifo) =>
                fifo.enq(value, en)
            }
            central_output_indices.enq(I32(ind), en)
        }

        // check break
        utils.MaybeRead(arbiter_command, arbiter_break)
      }
    }

    val processor_break: Reg[Bit] = Reg[Bit](false, "processor_break")
    'Coprocessor.Stream(breakWhen = processor_break).Foreach(*) {
      _ =>
        utils.checkpoint("CoprocActive")
        val empty_queue = central_input_fifos map {
          _.isEmpty
        } reduceTree {
          _ || _
        }
        ifThenElse(
          empty_queue, () => {
            // Do nothing
          }, () => {
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
        )

        // check on break
        utils.MaybeRead(processor_command, processor_break)
    }
  }

  // Process function takes an input read from a fifo and writes to the corresponding output fifo.

  // Interface for "using" the coprocessor
  def apply(input: Seq[In_T]): Seq[Out_T] = {
    assert(!frozen)
    // create new input and output queues
    println(s"Inputs: ${input.mkString(", ")}")
    assert(input.size == input_arity, f"Expected Input Arity:  $input_arity, called with arity: ${input.size}")

    val new_input_fifo_set = input_fifos(cnt)
    val output_fifo_set = output_fifos(cnt)
    cnt += 1

    val result = Range(0, output_arity) map {_ => Reg[Out_T]}

    'CoprocessorInput.Stream {
      utils.checkpoint("PreEnqueue")

      (new_input_fifo_set zip input) foreach {
        case (fifo, in) => fifo.enq(in)
      }
    }

    'CoprocessorOutput.Stream {
      (output_fifo_set zip result) foreach {
        case (fifo, res) =>
          res := fifo.deq
      }
    }

    result map {_.value}
  }
}
