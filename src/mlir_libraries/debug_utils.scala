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
  private val scope_id = state.GetCurrentHandle()
  private val dumps = new ListBuffer[() => Unit]
  private val stores = new ListBuffer[() => Unit]
  private var sramScopeId: argon.BundleHandle = _

  private def escape[T](id: argon.BundleHandle)(thunk: => T): T = state.WithScope(id){thunk}

  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  def dump[T: Num](name: String)(arg: types.ReadableND[T]): types.ReadableND[T] = {

    val (validDram, dram, requestDram) = escape(scope_id) {
      val escapeDram = DRAM[T](arg.size)
      escapeDram.explicitName = f"dump_DRAM_$name"

      val validDram = DRAM[I32](arg.size)

      val requestDram = DRAM[I32](arg.size)
      (validDram, escapeDram, requestDram)
    }

    // banking params
    val N = arg.shape map {
      _ match {
        case argon.Const(x) => x.toInt
        case _ => assert(false, "FUCK")
          1
      }
    } reduce {_ * _}

    println(s"Banking by $N: ${arg.shape}")

    val intermediate = escape(sramScopeId) { SRAM[T](arg.size).nonbuffer.forcebank(Seq(N), Seq(1), Seq(1)) }
    intermediate.explicitName = f"dump_SRAM_$name"

    val accessSram = escape(sramScopeId) { SRAM[I32](arg.size).nonbuffer.forcebank(Seq(N), Seq(1), Seq(1)) }
    accessSram.explicitName = f"dump_SRAM_valid_$name"

    val requestSram = escape(sramScopeId) { SRAM[I32](arg.size).nonbuffer.forcebank(Seq(N), Seq(1), Seq(1)) }
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
            requestSram.write(1.to[I32], Seq(utils.computeIndex(index, strides)), ens)
            debug_utils.TagVector(s"dump_enq_$name", index, ens)
            subInterface.enq(index, ens)
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            val v = subInterface.deq(index, ens)
            debug_utils.TagVector(s"dump_deq_$name", index, ens)
            intermediate.write(v,Seq(utils.computeIndex(index, strides)), ens)
            accessSram.write(1.to[I32],Seq(utils.computeIndex(index, strides)), ens)
            v
          }
        }
      }

      override val shape: Seq[dsl.I32] = arg.shape
    }
  }

  def store: Unit = {
    println(s"Stores: $stores")
    escape(sramScopeId) {
      stores foreach { x => x() }
    }
  }

  def dump = {
    println(s"Dumps: $dumps")
    escape(scope_id) {
      dumps foreach { x => x() }
    }
  }

  def setSramScope = {
    sramScopeId = state.GetCurrentHandle()
  }
}
