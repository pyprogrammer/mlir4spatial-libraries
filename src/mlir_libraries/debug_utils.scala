package mlir_libraries
import spatial.libdsl._
import _root_.spatial.metadata.memory._
import _root_.spatial.dsl

import scala.collection.mutable.ListBuffer

object debug_utils {
  def tag_region[T: Bits](symbol: Symbol, tagged: T)(implicit state: argon.State) = {
    import spatial.dsl._
    val result = Reg[T]
    symbol.Pipe {
      result := tagged
    }
    result.value
  }

  def TagVector[T: Bits](name: String, values: Seq[T], ens: Set[Bit] = Set.empty)(implicit state: argon.State): Void = {
    if (!Options.Debug) { return new Void }
    {
      // For scala sim
      import spatial.dsl._
      printIf(ens, r"${name} =")
    }
    values.zipWithIndex foreach {
      case (value, index) =>
        val reg = Reg[T].dontTouch
        reg.explicitName = f"${name}_$index"
        reg := value

        {
          // For scala sim
          import spatial.dsl._
          printIf(ens, r" $value")
        }
    }

    {
      // For scala sim
      import spatial.dsl._
      printIf(ens, argon.lang.api.Text("\n"))
    }
  }
}

class DumpScope(implicit state: argon.State) {
  // current id
  val scope_id = state.bundleStack.size
  val dumps = new ListBuffer[() => Unit]
  val stores = new ListBuffer[() => Unit]

  private def escape[T](thunk: => T): T = {
    val bundle = state.bundleStack(scope_id)
    val (result, newBundle) = state.WithScope({
      val tmp = thunk
      tmp
    }, bundle)
    // store the new bundle back
    state.bundleStack.update(scope_id, newBundle)
    result
  }

  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }



  def dump[T: Num](name: String)(arg: types.ReadableND[T]): types.ReadableND[T] = {
    val (validDram, dram, requestDram) = escape {
      val escapeDram = DRAM[T](arg.size)
      escapeDram.explicitName = f"dump_DRAM_$name"

      val validDram = DRAM[I32](arg.size)

      val requestDram = DRAM[I32](arg.size)
      (validDram, escapeDram, requestDram)
    }

    val intermediate = SRAM[T](arg.size).nonbuffer.nofission
    intermediate.shouldIgnoreConflicts = true
    intermediate.explicitName = f"dump_SRAM_$name"

    val accessSram = SRAM[I32](arg.size).nonbuffer.nofission
    accessSram.shouldIgnoreConflicts = true
    accessSram.explicitName = f"dump_SRAM_valid_$name"

    val requestSram = SRAM[I32](arg.size).nonbuffer.nofission
    requestSram.shouldIgnoreConflicts = true
    requestSram.explicitName = f"dump_SRAM_rqst_$name"

    val strides = computeStrides(arg.shape)

    stores.append(() => {
      dram store intermediate
      validDram store accessSram
      requestDram store requestSram
    })

    dumps.append(() => {
      val mem = getMem(dram)
      writeCSV1D(mem, f"${name}.csv", delim = ",")

      val valid = getMem(validDram)
      writeCSV1D(valid, f"${name}_valid.csv", delim = ",")

      val requests = getMem(requestDram)
      writeCSV1D(requests, f"${name}_rqst.csv", delim = ",")
    })

    new types.ReadableND[T] {

      override def getInterface: types.Interface[T] = {
        val subInterface = arg.getInterface

        new types.Interface[T] {
          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {
            argon.stage(spatial.node.SRAMWrite(requestSram,1.to[I32],Seq(utils.computeIndex(index, strides)), ens))
            debug_utils.TagVector(s"dump_enq_$name", index, ens)
            subInterface.enq(index, ens)
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            val v = subInterface.deq(index, ens)
            debug_utils.TagVector(s"dump_deq_$name", index, ens)
            argon.stage(spatial.node.SRAMWrite(intermediate,v,Seq(utils.computeIndex(index, strides)), ens))
            argon.stage(spatial.node.SRAMWrite(accessSram,1.to[I32],Seq(utils.computeIndex(index, strides)), ens))
            v
          }
        }
      }

      override val shape: Seq[dsl.I32] = arg.shape
    }
  }

  def store: Unit = {
    println(s"Stores: $stores")
    stores foreach {x => x()}
  }

  def dump = {
    println(s"Dumps: $dumps")
    dumps foreach {x => x()}
  }
}
