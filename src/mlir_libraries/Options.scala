package mlir_libraries

object Options {
  def getOption(str: String, default: Boolean = false) = {
    val result = sys.env.getOrElse(str, if (default) "true" else "false") == "true"
    println(s"Option: $str = $result")
    result
  }

  val PO2Opt: Boolean = getOption("PO2Opt", true)
  val Coproc: Boolean = getOption("Coproc", true)
  val StreamLattice: Boolean = getOption("StreamLattice")

  val Debug: Boolean = getOption("Debug", false)
  val Verify: Boolean = getOption(str = "Verify", false)
}

trait OptionType[T] {
  def parse(string: String): T
}

class BoolOption extends OptionType[Boolean] {
  override def parse(string: String): Boolean = {
    string.toLowerCase match {
      case "true" | "1" | "t" => true
      case "false" | "0" | "f" => false
    }
  }
}
