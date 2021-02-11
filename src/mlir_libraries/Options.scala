package mlir_libraries

object Options {
  def getOption(str: String, default: Boolean = false) = {
    val result = sys.env.getOrElse(str, if (default) "true" else "false") == "true"
    println(s"Option: $str = $result")
    result
  }

  lazy val Debug: Boolean = getOption("Debug", false)
  lazy val Verify: Boolean = getOption(str = "Verify", false)
}
