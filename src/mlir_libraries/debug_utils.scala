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

  @forge.tags.api def TagVector[T: Bits](name: String, values: Seq[T], ens: Set[Bit] = Set.empty): Void = {
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
        reg.write(value, ens.toSeq:_*)

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

  case class SRAMDRAMPair[T: Num](sram: RegFile1[T], dram: DRAM1[T]) {
    def transfer() = {
      dram store sram
    }
  }
  case class DumpBundle[T: Num](valid: SRAMDRAMPair[I32], request: SRAMDRAMPair[I32], values: SRAMDRAMPair[T])

  def dump[T: Num](name: String)(arg: types.ReadableND[T]): types.ReadableND[T] = {
    if (!Options.Debug) {
      return arg
    }

    val bundles = scala.collection.mutable.ListBuffer[DumpBundle[T]]()

    val strides = computeStrides(arg.shape)

    stores.append(() => {
      println(s"Storing: ${bundles.mkString(", ")}")
      bundles foreach {
        bundle =>
          bundle.request.transfer()
          bundle.valid.transfer()
          bundle.values.transfer()
      }
    })

    dumps.append(() => {
      println(s"Dumping: ${bundles.mkString(", ")}")
      val output = spatial.dsl.Array.empty[T](arg.size)
      val outputValid = spatial.dsl.Array.empty[Bit](arg.size)

      Foreach(0 until arg.size) {
        i => outputValid(i) = false
      }

      bundles.zipWithIndex foreach {
        case (bundle, ind) =>
          // assert that for each request there was a response
          val rqst = getMem(bundle.request.dram)
          val valid = getMem(bundle.valid.dram)
          val filled = rqst.zip(valid) { case (a, b) => (a !== I32(0)) && (b !== I32(0))}

          {
            import spatial.dsl._
            Foreach(0 until arg.size) {
              i =>
                assert(rqst(i) === valid(i), r"Every Request must have a Response, mismatched at $name ${i} Rqst: ${rqst(i)}, Resp: ${valid(i)}")
            }

            val results = getMem(bundle.values.dram)
            Foreach(0 until arg.size) {
              i =>
                ifThenElse(filled(i), () => {
                  ifThenElse(outputValid(i), () => {
                    assert(abs(output(i) - results(i)) < (0.001).toUnchecked[T], r"Previously received ${output(i)}, now receiving ${results(i)} at $name ${i}")
                  }, () => {
                    output(i) = results(i)
                    outputValid(i) = true
                  })
                }, () => {})
            }
          }

          writeCSV1D(rqst, f"${name}_rqst_$ind.csv", delim = ",")
          writeCSV1D(valid, f"${name}_valid_$ind.csv", delim = ",")
      }

      writeCSV1D(output, f"${name}.csv", delim = ",")
      writeCSV1D(outputValid, f"${name}_valid.csv", delim = ",")
    })

    new types.ReadableND[T] {

      override def getInterface: types.Interface[T] = {
        val subInterface = arg.getInterface

        val (validDram, valueDram, requestDram) = escape(scope_id) {
          val valueDram = DRAM[T](arg.size)
          valueDram.explicitName = f"dump_DRAM_$name"

          val validDram = DRAM[I32](arg.size)

          val requestDram = DRAM[I32](arg.size)
          (validDram, valueDram, requestDram)
        }

        val (validSram, valueSram, requestSram) = escape(sramScopeId) {

          val zeros = arg.size match {
            case argon.Const(v) => Some(Range(0, v.toInt) map {_ => I32(0)})
            case _ => None
          }

          val intermediate = RegFile[T](arg.size)
          intermediate.explicitName = f"dump_SRAM_$name"
          intermediate.dontTouch
          intermediate.nonbuffer

          val accessSram = RegFile[I32](arg.size, zeros)
          accessSram.explicitName = f"dump_SRAM_valid_$name"
          accessSram.dontTouch
          accessSram.nonbuffer

          val requestSram = RegFile[I32](arg.size, zeros)
          requestSram.explicitName = f"dump_SRAM_rqst_$name"
          requestSram.dontTouch
          requestSram.nonbuffer

          (accessSram, intermediate, requestSram)
        }
        bundles.append(
          DumpBundle(
            SRAMDRAMPair(validSram, validDram),
            SRAMDRAMPair(requestSram, requestDram),
            SRAMDRAMPair(valueSram, valueDram)
          )
        )

        new types.Interface[T] {
          override def enq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): Void = {
            requestSram.write(1.to[I32], Seq(utils.computeIndex(index, strides)), ens)
            debug_utils.TagVector(s"dump_enq_$name", index, ens)
            subInterface.enq(index, ens)
          }

          override def deq(index: Seq[dsl.I32], ens: Set[dsl.Bit]): T = {
            val v = subInterface.deq(index, ens)
            val linearizedIndex = Seq(utils.computeIndex(index, strides))
            debug_utils.TagVector(s"dump_deq_$name", index, ens)
            debug_utils.TagVector(s"dump_value_$name", Seq(v), ens)
            valueSram.write(v, linearizedIndex, ens)
            validSram.write(1.to[I32], linearizedIndex, ens)
            v
          }
        }
      }

      override val shape: Seq[dsl.I32] = arg.shape
    }
  }

  def store(): Unit = {
    println(s"Stores: $stores")
    stores foreach { x => x() }
  }

  def dump() = {
    println(s"Dumps: $dumps")
    dumps foreach { x => x() }

  }

  def setSramScope = {
    sramScopeId = state.GetCurrentHandle()
  }
}
