plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

val ktor_version = "3.1.1"

application {
    applicationName = "hp3tri"
    mainClass.set("dev.ckozel.alife.hp3tri.MainKt")
    val jepPath: String
    val libraryPath: String
    if (org.gradle.internal.os.OperatingSystem.current().isWindows) {
        jepPath = "${project.rootDir}/.venv/Lib/site-packages/jep"
        // python312.dll must be on java.library.path for jep.dll to load
        val pythonHome = File("${project.rootDir}/.venv/Scripts/python.exe").canonicalFile
            .parentFile.parentFile.let { venvDir ->
                // Resolve base Python install from venv's pyvenv.cfg
                val cfg = File(venvDir, "pyvenv.cfg")
                if (cfg.exists()) {
                    cfg.readLines().firstOrNull { it.startsWith("home") }
                        ?.substringAfter("=")?.trim() ?: venvDir.absolutePath
                } else venvDir.absolutePath
            }
        libraryPath = "$jepPath;$pythonHome"
    } else {
        jepPath = "${project.rootDir}/.venv/lib/python3.12/site-packages/jep"
        libraryPath = jepPath
    }
    applicationDefaultJvmArgs = listOf(
        "-Djava.library.path=$libraryPath",
    )
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")

    implementation("io.ktor:ktor-server-core:$ktor_version")
    implementation("io.ktor:ktor-server-netty:$ktor_version")
    implementation("io.ktor:ktor-server-content-negotiation:$ktor_version")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktor_version")
    implementation("io.ktor:ktor-server-cors:$ktor_version")

    implementation("org.slf4j:slf4j-simple:2.0.16")

    // Jep - Java Embedded Python
    implementation("black.ninia:jep:4.3.1")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.3")
}

tasks.test {
    useJUnitPlatform()
}

tasks.named<JavaExec>("run") {
    workingDir = rootDir
    val venvSitePackages: String
    if (org.gradle.internal.os.OperatingSystem.current().isWindows) {
        venvSitePackages = "${rootDir}/.venv/Lib/site-packages"
        // Windows DLL loader finds python312.dll via PATH, not java.library.path
        val pythonHome = File("${rootDir}/.venv/pyvenv.cfg").let { cfg ->
            if (cfg.exists()) {
                cfg.readLines().firstOrNull { it.startsWith("home") }
                    ?.substringAfter("=")?.trim() ?: ""
            } else ""
        }
        if (pythonHome.isNotEmpty()) {
            val currentPath = System.getenv("PATH") ?: ""
            environment("PATH", "$pythonHome;$currentPath")
        }
    } else {
        venvSitePackages = "${rootDir}/.venv/lib/python3.12/site-packages"
    }
    environment("PYTHONPATH", venvSitePackages)
    environment("TAICHI_ARCH", System.getenv("TAICHI_ARCH") ?: "cuda")
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}
