package mlir_libraries

case class Fraction(num: Int, den: Int) {
}

object Fraction {
  def fromInt(num: Int): Fraction = Fraction(num, 1)
}
