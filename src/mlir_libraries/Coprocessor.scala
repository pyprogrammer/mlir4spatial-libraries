package mlir_libraries

import forge.tags.api
import spatial.libdsl._
import spatial.metadata.memory._
import argon.State
import mlir_libraries.Coprocessor.getId

class CoprocessorScope(val coprocessorScopeId: argon.BundleHandle, val setupScopeId: argon.BundleHandle, killReg: Option[Reg[Bit]] = None)(implicit s: argon.State) {
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

  private def escapeToScope[T](id: argon.BundleHandle, thunk: => T): T = {
    state.WithScope(id){
      val tmp = thunk
      tmp
    }
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
    val setupScopeId = state.GetCurrentHandle()
    val coprocScopeId = state.GetCurrentHandle()
    if (Options.Coproc) {
      val kill: Reg[Bit] = Reg[Bit](false, "CoprocessorScopeKill").dontTouch
      'CoprocessorScope.Stream(breakWhen = kill)(implicitly, implicitly) {
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

  protected val INPUT_FIFO_DEPTH = 32
  protected val OUTPUT_FIFO_DEPTH = 32
  protected val PREALLOC_CREDITS = OUTPUT_FIFO_DEPTH / 2
  protected val DELAYED_CREDITS = OUTPUT_FIFO_DEPTH - PREALLOC_CREDITS - 2

  private val CREDIT_REPLICANTS = 16

  protected val id = getId

  coprocessorScope.register(instantiate)

  // Need to construct a new type and FIFO thingy.
  protected val input_fifos = collection.mutable.Buffer[FIFO[TaggedInput[In_T]]]()
  protected val output_fifos = collection.mutable.Buffer[FIFO[Out_T]]()
  protected val credit_fifos = collection.mutable.Buffer[FIFO[I32]]()

  var frozen: Boolean = false

  def enq(input: In_T, ens: Set[Bit]): Unit

  def deq(inputs: In_T, ens: Set[Bit]): Out_T

  def numInterfaces = {
    assert(frozen, "Must be frozen before getting the number of interfaces")
    input_fifos.size
  }

  def instantiate(): Void = {
    assert(!frozen, "Shouldn't be frozen yet")
    frozen = true

    assert(input_fifos.nonEmpty, "Input Fifo's cant be empty!")
    assert(output_fifos.size == numInterfaces, s"Expected $numInterfaces fifos, got ${output_fifos.size}")
    assert(credit_fifos.size == numInterfaces, s"Expected $numInterfaces fifos, got ${credit_fifos.size}")

    val central_input_fifo = FIFO[TaggedInput[In_T]](I32(INPUT_FIFO_DEPTH))
    central_input_fifo.explicitName = s"CentralInputFifo${id}"

    // Continuously pump this fifo full to keep arbiter running
    val prealloc_fifo = FIFO[I32](I32(4))

    // This is so that it can start running.
    'CoprocessorBusyStuffing.Pipe.Foreach(DELAYED_CREDITS by 1) {
      _ =>
        Foreach(input_fifos.size by 1) {
          ind => prealloc_fifo.enq(ind)
        }
    }

    val credits = RegFile[I32](I32(CREDIT_REPLICANTS), I32(numInterfaces), Range(0, CREDIT_REPLICANTS * numInterfaces) map {_ => I32(PREALLOC_CREDITS / CREDIT_REPLICANTS)})
    credits.nonbuffer
    credits.explicitName = s"CreditRegFile_${id}"

    'CoprocessorArbiter.Pipe.Foreach(*) {
      iter => {
        val priorityDeqEnableFIFOs = Range(0, numInterfaces) map { i =>
          val v = Reg[Bit](false)
          v.explicitName = s"PriorityDeqEnableFIFO_${id}_${i}"
          v
        }
        'CoprocessorArbiterEnableCalcs.Pipe {
          val creditIter = iter % I32(CREDIT_REPLICANTS)
          Range(0, numInterfaces) foreach {
            iid =>
              priorityDeqEnableFIFOs(iid) := (credits(creditIter, I32(iid)) > I32(0))
          }
        }

        'CoprocessorArbiterSubEnqs.Pipe.II(1) {
          val allCreditFifos = credit_fifos ++ Seq(prealloc_fifo)
          val creditIter = iter % I32(CREDIT_REPLICANTS)
          val creditUpdate = priorityDeq(allCreditFifos: _*)
          val priorityDeqEnables = priorityDeqEnableFIFOs map {
            _.value
          }
          val isValid = priorityDeqEnables reduceTree {
            _ || _
          }
          val nextTask: TaggedInput[In_T] = if (Options.RoundRobin) {
            roundRobinDeq(input_fifos.toList, priorityDeqEnables.toList, iter)
          } else {
            priorityDeq(input_fifos.toList, priorityDeqEnables.toList)
          }
          // update the appropriate credit reg

          Range(0, numInterfaces) foreach {
            iid =>
              val isNextTask = I32(iid) === nextTask.id
              val receivedCredit = I32(iid) === creditUpdate
              val update = receivedCredit.to[I32] - isNextTask.to[I32]
              credits(creditIter, I32(iid)) = credits(creditIter, I32(iid)) + update
          }
          central_input_fifo.enq(nextTask, isValid)
          enq(nextTask.payload, Set(isValid))
        }
      }
    }

    val flushFIFO = FIFO[TaggedInput[In_T]](I32(8))
    flushFIFO.explicitName = "flushFifo"
    'CoprocessorDriver.Foreach(*) {
      _ =>
        val blank = TaggedInput(I32(-1), Bits[In_T].zero)
        flushFIFO.enq(blank)
    }

    val validFifo = FIFO[Bit](I32(OUTPUT_FIFO_DEPTH))
    val outputInfoFifo = FIFO[TaggedInput[In_T]](I32(OUTPUT_FIFO_DEPTH))
    'CoprocessorKernelEnable.Pipe.Foreach(*) {
      _ =>
        val outputInfo = priorityDeq(central_input_fifo, flushFIFO)
        val destination = outputInfo.id
        validFifo.enq(destination !== I32(-1))
        outputInfoFifo.enq(outputInfo)
    }

    'CoprocessorKernelMain.Pipe.Foreach(*) {
      _ =>
        val valid = validFifo.deq()
        val outputInfo = outputInfoFifo.deq()
        val results = deq(outputInfo.payload, Set(valid))
        output_fifos.zipWithIndex foreach {
          case (output_fifo, output_index) =>
            val write_enable = I32(output_index) === outputInfo.id
            output_fifo.enq(results, write_enable)
        }
    }
  }

  class CoprocessorInterface(input_stream: FIFO[TaggedInput[In_T]], output_stream: FIFO[Out_T], credit_stream: FIFO[I32], id: Int) {

    def enq(input: In_T, en: Bit = Bit(true)): Void = {
      val bundle = TaggedInput[In_T](I32(id), input)
      input_stream.enq(bundle, en)
    }

    def deq(en: Bit = Bit(true)): Out_T = {
      val v = output_stream.deq(en)
      credit_stream.enq(I32(id), en)
      v
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
      new_input_fifo.noduplicate
      new_input_fifo.explicitName = f"InputFIFO_${id}_${iid}"
      input_fifos.append(new_input_fifo)

      val output_fifo = FIFO[Out_T](I32(OUTPUT_FIFO_DEPTH))
      output_fifo.noduplicate
      output_fifo.explicitName = f"OutputFIFO_${id}_${iid}"

      output_fifos.append(output_fifo)

      val credit_fifo = FIFO[I32](I32(OUTPUT_FIFO_DEPTH))
      credit_fifo.noduplicate
      credit_fifo.explicitName = f"CreditFIFO_${id}_${iid}"

      credit_fifos.append(credit_fifo)

      (new_input_fifo, output_fifo, credit_fifo)
    }
    new CoprocessorInterface(io._1, io._2, io._3, iid)
  }
}
