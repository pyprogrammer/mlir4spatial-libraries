package mlir_libraries

import spatial.libdsl._
import _root_.spatial.libdsl
import _root_.spatial.metadata.memory._

object CommonConstructs {

  private def computeStrides[T: Num](shape: Seq[T])(implicit state: argon.State): Seq[T] = {
    val strides = shape.scanRight(1.to[T]) {
      case (s, stride) =>
        s * stride
    }
    strides.drop(1)
  }

  def Materialize[T: Num](arg: types.ReadableND[T])(implicit state: argon.State): types.ReadableND[T] = {
    val size = arg.shape reduceTree {
      _ * _
    }
    val intermediate = SRAM[T](size)
    val strides = computeStrides(arg.shape)

    val ctrs = arg.shape map { x => Counter.from(x by I32(1)) }

    Foreach(ctrs) {
      nd_index => {
        val index = utils.computeIndex(nd_index, strides)
        intermediate(index) = arg(nd_index: _*)()
      }
    }

    new types.ReadableND[T] {
      override lazy val shape = arg.shape

      override def apply(index: spatial.dsl.I32*): () => T = {
        () => {
          val ind = utils.computeIndex(index, strides)
          intermediate(ind)
        }
      }
    }
  }

//  def CoprocessorStage[T: Num](arg: types.ReadableND[T], workers: scala.Int, users: scala.Int = 64 /* Most of these will be unused */)(implicit state: argon.State, coprocessorScope: CoprocessorScope): types.ReadableND[T] = {
//    // Wraps the readable inside of a coprocessor.
//    val prealloc = workers / users + (users.toDouble / workers).ceil.toInt
//
//    val coprocessors = Range(0, workers) map { _ =>
//      new Coprocessor[I32, T](arg.shape.size, 1, prealloc) {
//        override def coprocessorScope: CoprocessorScope = coprocessorScope
//        override def execute(inputs: Seq[libdsl.I32]): Seq[T] = {
//          Seq(arg(inputs: _*)())
//        }
//      }
//    }
//  }
}
