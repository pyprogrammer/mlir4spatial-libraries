package mlir_libraries

case class OptimizationConfig(lattice_loops: Int = 0, pwl_iterations: Int = 1)


object LatticeOptions {

  sealed trait LatticeConfig
  case object Unrolled extends LatticeConfig
  case class Streamed(loop_dimensions: Int) extends LatticeConfig
  case class Recursive(loop_dimensions: Int) extends LatticeConfig
  case class Flattened(loop_dimensions: Int) extends LatticeConfig

}
