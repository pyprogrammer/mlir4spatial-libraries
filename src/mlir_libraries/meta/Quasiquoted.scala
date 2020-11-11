package mlir_libraries.meta
import scala.language.implicitConversions

trait Quasiquoted[T] {
  def eval: T
}

object Quasiquoted {
  implicit def cast[T](quasiquoted: Quasiquoted[T]): T = quasiquoted.eval
  implicit def doubleCast[U, T](quasiquoted: Quasiquoted[T])(implicit conv: T => U): U = conv(quasiquoted.eval)
}
