# Singleton Design Pattern
- A singleton class only has one instance and provides a global point of access to it.
- Used when we want a centralised control of resources, such as managing db connections, configuration settings or logging.

**General Description**: Memory holds a single instance of the Singleton class. There is a private static method where the class is instantiated, which is the private constructor. A public static method, called something like Getinstance() either creates the singleton (if one doesn't exist using lazy initialisation), or returns the singleton class that currently exists.

# Factory Design Pattern
- A factory method is a design pattern that provides an interface for creating objects in a superclass, but allows subclasses to alter methods.
- The user will ask the factory for an object, and the factory decides which specific class to create.
- For example, imagine asking for "coffee" at a cafe, and the barista decides whether to serve a Latte, Long Black or Flate White.
- To simplify, a factory pattern is just a helper that makes objects for you. For example, you can ask for a circle, and the factory will instantiate the circle object and pass that back to you. Allows prevention of many if statements.