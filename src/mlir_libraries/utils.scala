package mlir_libraries

import spatial.libdsl._

object utils {
  implicit def convertToSpatialArray[T:Num](arg: scala.Array[T])(implicit state: argon.State): spatial.lang.Tensor1[T] = {
    Tensor1(arg:_*)
  }
}
