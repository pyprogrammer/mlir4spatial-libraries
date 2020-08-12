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
    val reg: Reg[Bit] = name match {
      case Some(name) =>
        Reg[Bit](0.to[Bit], s"ckpt_$name").dontTouch
      case None =>
        checkpoints += 1
        Reg[Bit](0.to[Bit], s"ckpt_$checkpoints").dontTouch
    }
    reg := 1.to[Bit]
  }

  def save_value[T: Bits](name: String, value: T)(implicit state: argon.State, srcCtx: SrcCtx): Void = {
    val saver = Reg[T](Bits[T].zero, name).dontTouch
    saver := value
    new Void
  }

  def MaybeRead[T: argon.Type](fifo: FIFO[T], reg: Reg[T])(implicit state: argon.State, srcCtx: SrcCtx) = {
    ifThenElse(
      fifo.isEmpty,
      () => {},
      () => {
        reg := fifo.deq()
      }
    )
  }
}
