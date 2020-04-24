name := "spatial-lattice"

version := "0.1"

scalaVersion := "2.12.7"

val paradise_version  = "2.1.0"
/** Macro Paradise **/
resolvers += Resolver.sonatypeRepo("snapshots")
resolvers += Resolver.sonatypeRepo("releases")
addCompilerPlugin("org.scalamacros" % "paradise" % paradise_version cross CrossVersion.full)

scalacOptions ++= Seq("-explaintypes")

//resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"

libraryDependencies ++= Seq(
  "edu.stanford.cs.dawn" %% "spatial" % "1.1-SNAPSHOT"
)
