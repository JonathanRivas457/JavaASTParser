# JavaASTParser
This project is an update of our previous ASTParser. New changes include
- Class names are now labeled using similarity function and openai API
- Classes from slf4j and JavaFX are now recognized and descriptions are given for them
- Functions are found and descriptions are given for them
- New input method
Input Method
- In order to run the parser on multiple ASTs, refer to the input.json file format. The file consists of a series of AST json files
  that will be parsed sequentially and at the end of the program the data is dumped into another json file (refer to data.json).
- Also note that you must download the documenation used to have the parser generate descriptions. Documentation can be downloaded from the documentation.
