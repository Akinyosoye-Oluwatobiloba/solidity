contract X {
    /// @notice E event
    /// @dev E event
    event E();
}

contract C {
    function g() public {
        emit X.E();
    }
}
