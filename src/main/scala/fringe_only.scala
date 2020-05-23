import spatial.dsl._

@spatial object ShellOnly extends SpatialApp {
  type T = spatial.dsl.FixPt[TRUE, _24, _8]
  val dimensions = 5
  val iterations = 8
  def main(args: Array[String]) : Unit = {
    val input_DRAM = DRAM[T](iterations, dimensions)
    val output_DRAM = DRAM[T](iterations)

    Accel {
      val input_sram = SRAM[T](iterations, dimensions)
      input_sram load input_DRAM(0 :: iterations, 0 :: dimensions)
      val output_sram = SRAM[T](iterations)
      Pipe.Foreach(iterations by 1) { i =>
        output_sram(i) = input_sram(i, 0)
      }

      output_DRAM store output_sram
    }
    println(getMem(output_DRAM))
  }
}
