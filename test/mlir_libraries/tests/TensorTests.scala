package mlir_libraries.tests

import org.scalatest.FunSuite

class TensorTests extends FunSuite {
  val tensor = mlir_libraries.Tensor(shape = Array(3, 5), values = Range(0, 15).toArray)

  test("StridesTest") {
    println(tensor.strides.mkString(", "))
  }

  test("BasicTensorGet") {
    assert(tensor(1, 2) == 7)
  }

  test("TransposedTensorGet") {
    assert(tensor.transpose()(2, 1) == 7)
  }

  test("To2D") {
    val twoDim = tensor.to2DSeq
    assert(twoDim(1)(2) == 7)
  }
}
