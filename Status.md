# Current Status 6/8/2026
Tested the tool loop
- No tool calls: Worked
- Single tool call with followup reasoning: Worked
**TODO:** 
- Test Images. Use hard coded image until passing
- Update tests

# 6/6/2026

Finished refactoring the tool loop (inital pass). 
**TODO:** Debug to test it and then update tests. 

Last session:
- Recognize if tools accept tool_call_id. Tool_call_id is part of ToolResult now.
- Simplify tool loop to rely on xai package 
- Create custom function to convert xai messages into a dict


