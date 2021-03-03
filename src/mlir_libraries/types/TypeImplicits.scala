package mlir_libraries.types

import forge.tags.stateful
import mlir_libraries.Tensor
import spatial.node.SRAMRead

object TypeImplicits {
  import spatial.libdsl._

  @stateful implicit def SRAMToND[T:Bits, C[U]](rm: SRAM[T,C])(implicit srcCtx: SrcCtx): ReadableND[T] = {
    new PureReadable[T] {
      override def execute(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]) = {
        assert(index.length == shape.length, f"Cannot read from a ReadableND converted to ND: ${index}, $srcCtx")
        argon.stage(SRAMRead(rm, index, ens))
      }
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  @stateful implicit def RegFileToND[T:Bits, C[U]](rm: RegFile[T, C])(implicit srcCtx: SrcCtx): ReadableND[T] = {
    new PureReadable[T] {
      override def execute(index: Seq[spatial.dsl.I32], ens: Set[spatial.dsl.Bit]) = {
        assert(index.length == shape.length, f"Cannot read from a ReadableND converted to ND: ${index}, $srcCtx")
        argon.stage(spatial.node.RegFileRead(rm, index, ens))
      }
      override lazy val shape: Seq[I32] = rm.dims
    }
  }

  def ArrayToTensor[T](array: Array[T]): Tensor[T] =
    Tensor(values = array, shape = Array(array.length))
}
