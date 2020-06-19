package mlir_libraries

import spatial.libdsl._
import mlir_libraries.ConversionImplicits._

object CommonConstructs {
  def Materialize[T:Num](arg: types.Readable2D[T])(implicit state: argon.State): types.Readable2D[T] = {
    val intermediate = SRAM[T](arg.shape.head, arg.shape(1))
    Foreach(arg.shape.head by I32(1), arg.shape(1) by I32(1)) {
      (i, j) =>
        intermediate(i, j) = arg(i, j)()
    }
    intermediate
  }
}
