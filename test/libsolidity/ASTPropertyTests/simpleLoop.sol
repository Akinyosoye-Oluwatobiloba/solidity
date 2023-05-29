contract C {
  function f() public pure {
     /// TestCase1: condition.operator
     for(uint i = 0; i < 42; ++i) {
     }
     /// TestCase2: initializationExpression.initialValue.value
     for(uint i = 1; i < 42; i = i * 2) {
     }
  }
}
// ----
// TestCase1: <
// TestCase2: 1
