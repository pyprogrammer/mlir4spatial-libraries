package mlir_libraries

object LatticeOptions {
  sealed trait LatticeImplMethod
  case object Unrolled extends LatticeImplMethod
  case class Streamed(loop_dimensions: Int) extends LatticeImplMethod
  case class Recursive(loop_dimensions: Int) extends LatticeImplMethod
  case class Flattened(loop_dimensions: Int) extends LatticeImplMethod

  case class PO2Opt(en: Boolean)
}

case class LatticeConfig(impl: LatticeOptions.LatticeImplMethod = LatticeOptions.Unrolled, PO2Opt: LatticeOptions.PO2Opt = LatticeOptions.PO2Opt(true))

object CoprocOptions {
  sealed trait CoprocMode
  case object Stream extends CoprocMode
  case object Pipelined extends CoprocMode
}

case class CoprocConfig(mode: CoprocOptions.CoprocMode = CoprocOptions.Stream)
