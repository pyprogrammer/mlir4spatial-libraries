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
    val (validDram, dram) = escape {
      val escapeDram = DRAM[T](arg.size)
      escapeDram.explicitName = f"dump_DRAM_$name"

      val validDram = DRAM[I32](arg.size)
      (validDram, escapeDram)
    }

    val intermediate = SRAM[T](arg.size).nonbuffer.nofission
    intermediate.shouldIgnoreConflicts = true
    intermediate.explicitName = f"dump_SRAM_$name"

    val accessSram = SRAM[I32](arg.size).nonbuffer.nofission
    accessSram.shouldIgnoreConflicts = true
    accessSram.explicitName = f"dump_SRAM_valid_$name"

    val strides = computeStrides(arg.shape)

    stores.append(() => {
      dram store intermediate
      validDram store accessSram
    })

    dumps.append(() => {
      val mem = getMem(dram)
      writeCSV1D(mem, f"${name}.csv", delim = ",")

      val valid = getMem(validDram)
      writeCSV1D(valid, f"${name}_valid.csv", delim = ",")
    })

    new types.ReadableND[T] {
      override def apply(index: Seq[dsl.I32], ens: Set[dsl.Bit]): () => T = {
        val tmp = arg(index, ens)

        () => {
          val result = tmp()
          argon.stage(spatial.node.SRAMWrite(intermediate,result,Seq(utils.computeIndex(index, strides)), ens))
          argon.stage(spatial.node.SRAMWrite(accessSram,1.to[I32],Seq(utils.computeIndex(index, strides)), ens))
          result
        }
      }

      override val shape: Seq[dsl.I32] = arg.shape
    }
  }

  def store: Unit = {
    stores foreach {x => x()}
  }

  def dump = {
    dumps foreach {x => x()}
  }
}
