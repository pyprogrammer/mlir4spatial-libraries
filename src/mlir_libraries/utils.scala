package mlir_libraries

import spatial.libdsl._

object utils {
  implicit def convertToSpatialArray[T:Num](arg: scala.Array[T])(implicit state: argon.State): spatial.lang.Tensor1[T] = {
    Tensor1(arg:_*)
  }

  def computeIndex(index: Seq[I32], strides: Seq[I32])(implicit state: argon.State, srcCtx: SrcCtx): I32 = {
    ((index zip strides) map { case (a, b) => a * b}) reduceTree {_ + _}
  }

  def checkpoint(implicit state: argon.State, srcCtx: SrcCtx): Void = {
    checkpoint(None)(state, srcCtx)
  }

  def checkpoint(name: String)(implicit state: argon.State, srcCtx: SrcCtx): Void = {
    checkpoint(Some(name))
  }

  private var checkpoints = -1
  def checkpoint(name: Option[String])(implicit state: argon.State, srcCtx: SrcCtx): Void = {
    if (!mlir_libraries.Options.Debug) { return new Void() }
    val reg: Reg[Bit] = name match {
      case Some(name) =>
        Reg[Bit](0.to[Bit], s"ckpt_$name").dontTouch
      case None =>
        checkpoints += 1
        Reg[Bit](0.to[Bit], s"ckpt_$checkpoints").dontTouch
    }
    reg := 1.to[Bit]
  }

  def ComputeStrides(dimensions: IndexedSeq[Int]): IndexedSeq[Int] = {
    val strides: scala.Array[Int] = scala.Array.fill(dimensions.length) {
      1
    }
    scala.Range(1, dimensions.length, 1) foreach {
      d => {
        strides(d) = strides(d - 1) * dimensions(d - 1)
      }
    }
    strides.reverse
  }

  def CartesianProduct[T](components: Seq[T]*): Seq[Seq[T]] = {
    components match {
      case Seq() => Seq.empty[Seq[T]]
      case Seq(one) => one map {Seq(_)}
      case first +: tail =>
        val sub = CartesianProduct(tail:_*)
        first flatMap {
          head =>
            sub map {
              part => Seq(head) ++ part
            }
        }
    }
  }

  // Takes a list of FIFOs, and produces an output FIFO which yields elements from those fifos.
  // Produces a func which yields an element from one of those fifos.
//  def mergeChain[T:Bits](fifos: Seq[FIFO[T]], enFIFO: Option[FIFO[Vec[Bit]]], numFIFOsPerStage: Int = 4)(implicit state: argon.State, srcCtx: SrcCtx): Unit => T = {
//    if (fifos.size <= numFIFOsPerStage) {
//      val outputFIFO = FIFO[T](I32(4))
//      Pipe.Foreach(*) {
//        _ =>
//          enFIFO match {
//            case Some(enf) =>
//              assert(fifos.size == enf.A.nbits)
//              val enable = enf.deq
//
//          }
//      }
//    }
//  }
}
