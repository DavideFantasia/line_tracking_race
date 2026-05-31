cmake_minimum_required(VERSION 3.5)

project(line_tracking_race_application)

find_package(ament_cmake REQUIRED)

# per mettere lo script nell'eseguibile
install(PROGRAMS
  GeneticLineTracker.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
