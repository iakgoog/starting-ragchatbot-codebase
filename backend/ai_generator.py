from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Content Search Tool** (`search_course_content`): For specific course content and detailed educational materials
2. **Course Outline Tool** (`get_course_outline`): For course structure, lesson lists, and course overview information

Tool Usage Guidelines:
- **Course outline queries**: Use the outline tool for questions about course structure, lesson lists, what topics are covered, or course overview
- **Content-specific queries**: Use the search tool for detailed questions about specific topics, concepts, or materials within courses
- **Sequential tool usage**: You can use tools sequentially to build comprehensive answers (maximum 2 rounds)
- **Multi-step reasoning**: Use course outline first to understand structure, then search for specific content when needed
- **Comparative analysis**: Get information from multiple sources to compare courses, lessons, or topics
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Multi-Step Tool Examples:
- "Compare lesson 4 of course X with similar content" → Get course outline first, then search for similar topics
- "Find prerequisites for advanced topics" → Search for topic details first, then find related foundational content
- "Get course overview then specific implementation" → Use outline tool, then targeted content search

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use outline tool first, then provide course title, course link, and complete lesson information
- **Course content questions**: Use search tool first, then answer based on results
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_tool_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_sequential_tool_execution(
                response, api_params, tool_manager, max_tool_rounds
            )

        # Return direct response
        return response.content[0].text

    def _handle_sequential_tool_execution(
        self,
        initial_response,
        base_params: Dict[str, Any],
        tool_manager,
        max_rounds: int = 2,
    ):
        """
        Handle sequential execution of tool calls across multiple rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool calling rounds

        Returns:
            Final response text after all tool execution rounds
        """
        # Initialize conversation history
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        # Sequential tool calling loop
        while round_count < max_rounds and current_response.stop_reason == "tool_use":
            round_count += 1

            # Execute single tool round
            messages, tool_success = self._execute_single_tool_round(
                messages, current_response, tool_manager
            )

            # If tool execution failed, break the loop
            if not tool_success:
                break

            # Prepare API call for next round
            round_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }

            # Add tools only if we haven't reached max rounds
            if round_count < max_rounds:
                round_params["tools"] = base_params.get("tools", [])
                round_params["tool_choice"] = {"type": "auto"}

            # Get next response
            current_response = self.client.messages.create(**round_params)

        # If we exited because of max rounds and still have tool_use, make final call without tools
        if current_response.stop_reason == "tool_use" and round_count >= max_rounds:
            # Execute final tool round
            messages, _ = self._execute_single_tool_round(
                messages, current_response, tool_manager
            )

            # Final API call without tools
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }

            current_response = self.client.messages.create(**final_params)

        # Return final response text
        return current_response.content[0].text

    def _execute_single_tool_round(
        self, messages: List[Dict], current_response, tool_manager
    ):
        """
        Execute tools for a single round and update conversation messages.

        Args:
            messages: Current conversation messages
            current_response: API response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, success_flag)
        """
        # Add AI's tool use response to conversation
        messages.append({"role": "assistant", "content": current_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        tool_success = True

        for content_block in current_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution error gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                        }
                    )
                    tool_success = False

        # Add tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages, tool_success
