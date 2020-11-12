package mlir_libraries

import forge.tags.api
import spatial.libdsl._
import _root_.spatial.metadata.memory._
import argon.State
import mlir_libraries.Coprocessor.getId

class CoprocessorScope(val coprocessorScopeId: scala.Int, val setupScopeId: scala.Int, killReg: Option[Reg[Bit]] = None)(implicit s: argon.State) {
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

  def kill(): Void = killReg match {
    case Some(reg) => reg := 1.to[Bit]
    case None => new Void
  }
//  def setup[T](thunk: => T): T = escapeToScope(setupScopeId, thunk)
}

object CoprocessorScope {
  def apply[T](init: CoprocessorScope => T)(func: (CoprocessorScope, T) => Any)(implicit state: argon.State): Void = {
    val setupScopeId: Int = state.bundleStack.size
    val coprocScopeId: Int = state.bundleStack.size
    println(s"BS Size: ${state.bundleStack.size}")
    if (Options.Coproc) {
      val kill: Reg[Bit] = Reg[Bit](false, "CoprocessorScopeKill")
      'CoprocessorScope.Stream(breakWhen = kill).Foreach(I32(1) by I32(1)) {
        _ =>
          // Stage into current scope
          val scope = new CoprocessorScope(coprocScopeId, setupScopeId, Some(kill))
          val initialized = init(scope)
          func(scope, initialized)
//          Pipe {
//            func(initialized)
//            kill := 1.to[Bit]
//          }
          scope.instantiate()
      }
    } else {
      'CoprocessorScope.Pipe {
        val scope = new CoprocessorScope(coprocScopeId, setupScopeId)
        val initialized = init(scope)
        func(scope, initialized)
      }
    }
  }
}

@argon.tags.struct
case class TaggedInput[T: Bits](id: I32, payload: T)

object Coprocessor {
  private var nextId = 0
  def getId = {
    val nid = nextId
    nextId += 1
    nid
  }
}

// The coprocessor scope defines the control stream.
abstract class Coprocessor[In_T: Bits, Out_T: Bits] {
  def coprocessorScope: CoprocessorScope
  implicit val state: argon.State = coprocessorScope.state

  type InT = In_T
  type OutT = Out_T

  protected val SCALE_FACTOR = 4
  protected val INPUT_FIFO_DEPTH = 128
  protected val OUTPUT_FIFO_DEPTH = 128
  protected val INITIAL_CREDITS = OUTPUT_FIFO_DEPTH - 2
  protected val DELAYED_CREDITS = 1
  protected val CREDIT_FIFO_DEPTH = 16
  protected val id = getId

  coprocessorScope.register(instantiate)

  // Need to construct a new type and FIFO thingy.
  protected val input_fifos = collection.mutable.Buffer[FIFO[TaggedInput[In_T]]]()
  protected val output_fifos = collection.mutable.Buffer[FIFO[Out_T]]()

  var frozen: Boolean = false

//  // Override this for the core inner function.
//  def execute(inputs: Seq[In_T]): Seq[Out_T]

  def enq(input: In_T): Unit

  def deq(inputs: In_T): Out_T

  def instantiate(): Void = {
    assert(!frozen, "Shouldn't be frozen yet")
    frozen = true

    assert(input_fifos.nonEmpty, "Input Fifo's cant be empty!")

    val central_input_fifo = FIFO[TaggedInput[In_T]](I32(INPUT_FIFO_DEPTH))
    central_input_fifo.explicitName = s"CentralInputFifo${id}"

//    val credit_fifo = FIFO[I32](I32(CREDIT_FIFO_DEPTH))

    // Initialize with DELAYED_CREDITS per input fifo
//    'CoprocessorCreditAllocator.Stream.Foreach(DELAYED_CREDITS by 1) {
//      _ =>
//        input_fifos.zipWithIndex foreach {
//          case (_, ind) => credit_fifo.enq(I32(ind))
//        }
//    }

    // now execute the actual kernel
    'CoprocessorArbiter.Stream.Foreach(*) {
      _ => {
        val ntReg1 = FIFO[TaggedInput[In_T]](I32(16))
        ntReg1.explicitName = s"NTReg1_${id}"
        val ntReg2 = FIFO[TaggedInput[In_T]](I32(16))
        ntReg2.explicitName = s"NTReg2_${id}"

        'CoprocessorArbiterSubEnqs.Pipe {
//          val credits = input_fifos map {_ => I32(INITIAL_CREDITS)}
          val nextTask = priorityDeq(input_fifos:_*)
          ntReg1.enq(nextTask)
          ntReg2.enq(nextTask)
        }
        'CoprocessorArbiterCentralInputEnq.Pipe {
          central_input_fifo.enq(ntReg2.deq)
        }
        'CoprocessorArbiterChildSignal.Pipe {
          enq(ntReg1.deq.payload)
        }
      }
    }

    'Coprocessor.Stream.Foreach(*) {
      _ =>
        Pipe {
          val outputInfo = central_input_fifo.deq
          val destination = outputInfo.id
          val results = deq(outputInfo.payload)
          //          credit_fifo.enq(destination)
          output_fifos.zipWithIndex foreach {
            case (output_fifo, output_index) =>
              val write_enable = I32(output_index) === destination
              output_fifo.enq(results, write_enable)
          }
        }
    }
  }

  // Process function takes an input read from a fifo and writes to the corresponding output fifo.

  class CoprocessorInterface(input_stream: FIFO[TaggedInput[In_T]], output_stream: FIFO[Out_T], id: Int) {
    var enqueued = false
    var dequeued = false

    def enq(input: In_T, en: Bit = Bit(true)): Void = {
      val bundle = TaggedInput[In_T](I32(id), input)
      input_stream.enq(bundle, en)
    }

    def deq(en: Bit = Bit(true)): Out_T = {
      output_stream.deq(en)
    }
  }

  private var interfaceId = 0
  def getInterfaceId = {
    val iid = interfaceId
    interfaceId += 1
    iid
  }

  def interface(implicit srcCtx: SrcCtx): CoprocessorInterface = {
    val iid = getInterfaceId
    println(s"Creating Interface: $id -> $iid")

    val io = coprocessorScope.escape {

      val new_input_fifo = FIFO[TaggedInput[In_T]](I32(INPUT_FIFO_DEPTH))
      new_input_fifo.explicitName = f"InputFIFO_${id}_${iid}"
      input_fifos.append(new_input_fifo)

      val output_fifo = FIFO[Out_T](I32(OUTPUT_FIFO_DEPTH))
      output_fifo.explicitName = f"OutputFIFO_${id}_${iid}"

      output_fifos.append(output_fifo)

      (new_input_fifo, output_fifo)
    }
    new CoprocessorInterface(io._1, io._2, iid)
  }
}
