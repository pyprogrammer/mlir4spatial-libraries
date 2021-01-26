package tensorflow_lattice
import mlir_libraries.types._
import mlir_libraries.utils.checkpoint
import spatial.libdsl._
import mlir_libraries.{Coprocessor, CoprocessorScope, OptimizationConfig, Tensor => MLTensor}
import _root_.spatial.metadata.memory._

trait Lattice {

  def Lattice[T: Num](lattice_kernel: MLTensor[scala.Double], tp: String, shape: MLTensor[scala.Int],
                       units: Int)(arg: ReadableND[T])(implicit state: argon.State, config: mlir_libraries.OptimizationConfig,
                                                       coprocessorScope: mlir_libraries.CoprocessorScope): ReadableND[T] = {
    val isFullyUnrolled = config.lattice_loops == 0

    val s = shape
    val lk = lattice_kernel
    val un = units
    val oc = config
    if (isFullyUnrolled) {
      val lattice = new FullyUnrolledLattice {
        override val shape: MLTensor[Int] = s
        override val lattice_kernel: MLTensor[Double] = lk
        override val units: Int = un
        override implicit val config: OptimizationConfig = oc
      }
      lattice(arg)
    } else {
      val lattice = new ReduceBasedLattice {
        override val shape: MLTensor[Int] = s
        override val lattice_kernel: MLTensor[Double] = lk
        override val units: Int = un
        override implicit val config: OptimizationConfig = oc
      }
      lattice(arg)
    }
  }
}

private object HypercubeLattice {
  val PO2Opt = sys.env.getOrElse("PO2Opt", "true") == "true"

  def allCorners(maxes: Seq[scala.Int], partials: Seq[Seq[scala.Int]] = Seq(Seq.empty)): Seq[Seq[scala.Int]] = maxes match {
    case Nil => Nil
    case h :: tail if tail.nonEmpty => (0 to h).flatMap { i => allCorners(tail, partials.map(_ ++ Seq(i))) }
    case h :: tail if tail.isEmpty => (0 to h).flatMap { i => partials.map(_ ++ Seq(i)) }
  }

  object CombinationTree {

    def apply[T](xs: Seq[T]*)(func: (T, T) => T): Seq[T] = combine(xs, func).head

    private def join[T](x1: Seq[T], x2: Seq[T], func: (T, T) => T): Seq[T] = x1.flatMap { a => x2.map { b => func(a, b) } }

    private def combine[T](xs: Seq[Seq[T]], func: (T, T) => T): Seq[Seq[T]] = {
      xs.length match {
        case 0 => throw new Exception("Empty reduction level:")
        case 1 => xs
        case len if len % 2 == 0 => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) }, func)
        case len => combine(List.tabulate(len / 2) { i => join(xs(2 * i), xs(2 * i + 1), func) } :+ xs.last, func)
      }
    }
  }

}
