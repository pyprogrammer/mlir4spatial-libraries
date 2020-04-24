package lattice

import emul.FixedPoint
import spatial.libdsl._
import argon.{Cast, uconst}

import java.io.{BufferedWriter, FileWriter}
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

//
//object PreProcessor {
//
//  def processKeyPoints(tensor_flow_keypoint_filename: String, processed_keypoint_filename: String) = {
//
//
//    val output_file = new BufferedWriter(new FileWriter(processed_keypoint_filename))
//    val csv_writer = new CSVWriter(output_file, ',', CSVWriter.NO_QUOTE_CHARACTER)
//    var processed_keypoints = new ListBuffer[Array[String]]()
//
//    var prev_point = Array[String]()
//
//    val tf_keypoints = io.Source.fromFile(tensor_flow_keypoint_filename)
//    for (line <- tf_keypoints.getLines) {
//      val keypoint : Array[String] = line.split(",").map(_.trim)
//
//
//      var slope : Float = 0
//      if (prev_point.length == 0) {
//        prev_point = keypoint
//      }
//      else {
//        slope = (keypoint(1).toFloat - prev_point(1).toFloat) / (keypoint(0).toFloat - prev_point(0).toFloat)
//      }
//
//
//      processed_keypoints += Array(prev_point(0), keypoint(0), prev_point(1), slope.toString)
//
//      prev_point = keypoint
//
//    }
//
//    csv_writer.writeAll(processed_keypoints.asJava)
//    output_file.close()
//
//
//  }
//
//
//}


class Calibrator[IntBits <: INT[_], MantissaBits <: INT[_], OutputIntBits <: INT[_], Sign <: BOOL[_]]
(input_breakpoints: Seq[Double],
 output_values: Seq[Double],
 exact_matches: Map[Double, Double])
(implicit state: argon.State,
 tev: argon.Type[FixPt[Sign, IntBits, MantissaBits]],
 oev: argon.Type[FixPt[TRUE, OutputIntBits, MantissaBits]],
 c1: Cast[FixPt[Sign, IntBits, MantissaBits], FixPt[TRUE, OutputIntBits, MantissaBits]],
 c2: Cast[Double, FixPt[Sign, IntBits, MantissaBits]],
 c3: Cast[Double, FixPt[TRUE, OutputIntBits, MantissaBits]],
 c4: Cast[Int, FixPt[TRUE, OutputIntBits, MantissaBits]]
) {
  type T = FixPt[Sign, IntBits, MantissaBits]
  type OutputType = FixPt[TRUE, OutputIntBits, MantissaBits]


  def uevaluate(input: Any): OutputType = evaluate(input.asInstanceOf[T])

  // The TF Lattice code is really fragile when it comes to this, since it does exact matching.
  // Since we convert to FixPt, they really need to be exact. Thankfully, these are only really used
  // on the initial inputs, we don't have to maintain exact precision throughout.
  def evaluate(input: T): OutputType = {

    // If the input is an exact match, then that's what we return. This overrides the piecewise linear functionality.
    // This may be empty, in which case we get an empty Seq
    val exact = (exact_matches map {
      case (key, value) =>
        // === because == only gets rewritten for code with @spatial.
        val cond = uconst[T](key) === input
        (cond, mux(cond, value.to[OutputType], 0.to[OutputType]))
    }).toSeq // Spatial implicits like reduceTree only work on Seqs


    val edges = if (input_breakpoints.nonEmpty) Seq(
      mux(input < input_breakpoints.head.to[T], output_values.head.to[OutputType], 0.to[OutputType]),
      mux(input >= input_breakpoints.last.to[T], output_values.last.to[OutputType], 0.to[OutputType])
    ) else Seq.empty[OutputType]

    val pwl = if (input_breakpoints.nonEmpty) {
      (((input_breakpoints zip input_breakpoints.tail) zip (output_values zip output_values.tail)) map {
        case ((xs, xe), (ys, ye)) =>
          mux(
            xs.to[T] <= input && input < xe.to[T],
            (input - xs.to[T]).to[OutputType] * ((ye - ys) / (xe - xs)).to[OutputType],
            0.to[OutputType]
          )
      }) ++ Seq(
        mux(input < input_breakpoints.head.to[T], output_values.head.to[OutputType], 0.to[OutputType]),
        mux(input >= input_breakpoints.last.to[T], output_values.last.to[OutputType], 0.to[OutputType])
      ) reduceTree {
        _ + _
      }
    }
    else 0.to[OutputType]

    if (exact_matches.nonEmpty) {
      val exact_value = exact map {
        _._2
      } reduceTree {
        _ + _
      }
      val use_exact = exact map {
        _._1
      } reduceTree {
        _ | _
      }
      if (input_breakpoints.nonEmpty) {
        mux(use_exact, exact_value, pwl)
      } else {
        exact_value
      }
    } else {
      pwl
    }
  }
}

