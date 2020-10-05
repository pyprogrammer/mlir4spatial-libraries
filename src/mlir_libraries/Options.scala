package mlir_libraries

object Options {
  def getOption(str: String, default: Boolean = false) = {
    sys.env.getOrElse(str, if (default) "true" else "false") == "true"
  }

  lazy val PO2Opt: Boolean = getOption("PO2Opt", true)
  lazy val Coproc: Boolean = getOption("Coproc", true)

  lazy val Debug: Boolean = getOption("Debug", false)
  lazy val Verify: Boolean = getOption(str = "Verify", false)
}
