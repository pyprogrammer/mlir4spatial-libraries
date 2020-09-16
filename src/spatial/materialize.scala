package mlir_libraries

import spatial.libdsl._

// For MLIR-spatial native operations
object spatiallib {
  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  def Materialize[T: Num](parallelization: Int = 1)(arg: types.ReadableND[T])(implicit state: argon.State): types.ReadableND[T] = {
    val size = arg.shape reduceTree {
      _ * _
    }
    val intermediate = SRAM[T](size)
    val strides = computeStrides(arg.shape)

    val ctrs = arg.shape.zipWithIndex map { case(x, ind) => Counter.from(x by I32(1) par I32(if (ind == arg.shape.size - 1) parallelization else 1)) }

    Foreach(ctrs) {
      nd_index => {
        val index = utils.computeIndex(nd_index, strides)
        intermediate(index) = arg(nd_index, Set(Bit(true)))()
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      override def apply(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]): () => T = {
        () => {
          val ind = utils.computeIndex(index, strides)
          intermediate(ind)
        }
      }
    }
  }

  def CoprocessorStage[T: Num](arg: types.ReadableND[T], workers: scala.Int = 1)(implicit state: argon.State, cps: CoprocessorScope): types.ReadableND[T] = {

    val coprocessors = {
      Range(0, workers) map { _ =>
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
        println(s"Coprocessor Use: $count, assigned to ${count % workers}")
        val coprocessor = coprocessors(count % workers)
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
