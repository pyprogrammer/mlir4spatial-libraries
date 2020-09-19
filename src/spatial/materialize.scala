package mlir_libraries

import spatial.libdsl._
import spatial.metadata.memory._

// For MLIR-spatial native operations
object spatiallib {
  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  var materialization_cnt = -1
  def Materialize[T: Num](parallelization: Int = 1, uptime: Fraction = Fraction(1, 1))(arg: types.ReadableND[T])(implicit state: argon.State): types.ReadableND[T] = {
    materialization_cnt += 1
    val size = arg.shape reduceTree {
      _ * _
    }
    val intermediate = SRAM[T](size).nonbuffer
    intermediate.explicitName = f"materialization_sram_$materialization_cnt"
    val strides = computeStrides(arg.shape)

    val ctrs = arg.shape.zipWithIndex map { case(x, ind) => Counter.from(x by I32(1) par I32(if (ind == arg.shape.size - 1) parallelization else 1)) }

    Foreach(ctrs) {
      nd_index => {
        val index = utils.computeIndex(nd_index, strides)
        intermediate(index) = arg(nd_index, Set(Bit(true)))()
      }
    }

    retimeGate()

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      var metacnt = -1
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        metacnt += 1
        println(s"Materialize: $materialization_cnt $metacnt")
        () => {
          val ind = utils.computeIndex(index, strides)
          val tmp = intermediate(ind)
          val dbg_index = Reg[I32]
          dbg_index.dontTouch
          dbg_index := ind
          dbg_index.explicitName = f"materialization_dbg_${materialization_cnt}_${metacnt}"

          val retVal = Reg[T]
          retVal.dontTouch
          retVal.explicitName = f"materialization_retval_${materialization_cnt}_${metacnt}"
          retVal := tmp
          retVal
        }
      }
    }
  }

  def CoprocessorStage[T: Num](parallelization: scala.Int = 1, uptime: Fraction = Fraction(1, 1))(arg: types.ReadableND[T])(implicit state: argon.State, cps: CoprocessorScope): types.ReadableND[T] = {

    val coprocessors = {
      Range(0, parallelization) map { _ =>
        new Coprocessor[I32, T](arg.shape.size, 1) {
          override def coprocessorScope: CoprocessorScope = cps

          override def execute(inputs: Seq[I32]): Seq[T] = {
            Seq(arg(inputs, Set(Bit(true)))())
          }
        }
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      var count = 0
      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        println(s"Coprocessor Use: $count, assigned to ${count % parallelization}")
        val coprocessor = coprocessors(count % parallelization)
        count += 1
        val result = Reg[T]
        val interface = coprocessor.interface
        val en = ens.toSeq reduceTree {_ && _}
        ifThenElse(en, () => {
          Pipe {
            Stream {
              interface.enq(index)
            }
            Stream {
              result := interface.deq().head
            }
          }}, () => {})
        () => result.value
      }
    }
  }
}
