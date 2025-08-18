import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from search_tools import CourseSearchTool, CourseOutlineTool

class TestRAGSystem:
    """Test cases for RAG System integration"""
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_init_registers_both_tools(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test that RAGSystem registers both search and outline tools"""
        rag_system = RAGSystem(mock_config)
        
        # Verify both tools are registered
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
        
        # Verify tools are the correct types
        assert isinstance(rag_system.tool_manager.tools["search_course_content"], CourseSearchTool)
        assert isinstance(rag_system.tool_manager.tools["get_course_outline"], CourseOutlineTool)
        
        # Verify tool definitions are available
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_defs) == 2
        
        tool_names = [tool["name"] for tool in tool_defs]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_content_question_calls_search_tool(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test that content queries trigger the search tool"""
        rag_system = RAGSystem(mock_config)
        
        # Mock AI generator to simulate tool use
        mock_ai_gen.return_value.generate_response.return_value = "Answer about Python functions"
        
        # Mock search tool to return sources
        mock_sources = [{"text": "Python Course - Lesson 2", "link": "https://example.com"}]
        rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("How do Python functions work?")
        
        # Verify AI generator was called with tools
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        
        assert "tools" in call_args
        assert call_args["tool_manager"] == rag_system.tool_manager
        assert len(call_args["tools"]) == 2  # Both search and outline tools
        
        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
        
        assert response == "Answer about Python functions"
        assert sources == mock_sources
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_outline_question_calls_outline_tool(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test that outline queries trigger the outline tool"""
        rag_system = RAGSystem(mock_config)
        
        # Mock AI generator response for outline query
        mock_ai_gen.return_value.generate_response.return_value = "Course outline with 10 lessons"
        
        # Mock no sources for outline (outline tool doesn't track sources)
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("What lessons are in the Python course?")
        
        # Verify AI generator was called with tools
        mock_ai_gen.return_value.generate_response.assert_called_once()
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        
        assert "tools" in call_args
        assert call_args["tool_manager"] == rag_system.tool_manager
        
        assert response == "Course outline with 10 lessons"
        assert sources == []
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session_management(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test that session management works correctly"""
        rag_system = RAGSystem(mock_config)
        
        # Mock session manager
        mock_session_mgr.return_value.get_conversation_history.return_value = "Previous conversation"
        mock_session_mgr.return_value.add_exchange = Mock()
        
        # Mock AI response
        mock_ai_gen.return_value.generate_response.return_value = "AI response"
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("Test question", session_id="session_123")
        
        # Verify session history was retrieved
        mock_session_mgr.return_value.get_conversation_history.assert_called_once_with("session_123")
        
        # Verify conversation history was passed to AI
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous conversation"
        
        # Verify exchange was added to session
        mock_session_mgr.return_value.add_exchange.assert_called_once_with(
            "session_123", 
            "Test question", 
            "AI response"
        )
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test query without session ID"""
        rag_system = RAGSystem(mock_config)
        
        # Mock AI response
        mock_ai_gen.return_value.generate_response.return_value = "AI response"
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        response, sources = rag_system.query("Test question")
        
        # Verify no session operations were called
        mock_session_mgr.return_value.get_conversation_history.assert_not_called()
        mock_session_mgr.return_value.add_exchange.assert_not_called()
        
        # Verify no conversation history was passed
        call_args = mock_ai_gen.return_value.generate_response.call_args[1]
        assert call_args["conversation_history"] is None
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test course analytics functionality"""
        rag_system = RAGSystem(mock_config)
        
        # Mock vector store analytics
        mock_vector_store.return_value.get_course_count.return_value = 5
        mock_vector_store.return_value.get_existing_course_titles.return_value = [
            "Python Basics", "Advanced Python", "Web Development", "Data Science", "Machine Learning"
        ]
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Python Basics" in analytics["course_titles"]
        assert "Machine Learning" in analytics["course_titles"]
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_tool_execution_integration(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test that tools can actually be executed through the tool manager"""
        rag_system = RAGSystem(mock_config)
        
        # Test search tool execution
        search_result = rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="Python functions"
        )
        
        # Should call the vector store search method
        mock_vector_store.return_value.search.assert_called_once_with(
            query="Python functions",
            course_name=None,
            lesson_number=None
        )
        
        # Test outline tool execution  
        outline_result = rag_system.tool_manager.execute_tool(
            "get_course_outline",
            course_title="Python Basics"
        )
        
        # Should call vector store methods for course resolution and metadata
        mock_vector_store.return_value._resolve_course_name.assert_called()
        mock_vector_store.return_value.get_all_courses_metadata.assert_called()
    
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_tool_execution_with_invalid_tool(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_config):
        """Test tool execution with invalid tool name"""
        rag_system = RAGSystem(mock_config)
        
        result = rag_system.tool_manager.execute_tool("nonexistent_tool", param="value")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.path.isfile')
    @patch('rag_system.os.listdir')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_functionality(self, mock_doc_proc, mock_vector_store, mock_ai_gen, mock_session_mgr, mock_listdir, mock_isfile, mock_exists, mock_config):
        """Test adding course documents from folder"""
        rag_system = RAGSystem(mock_config)
        
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "ignore.jpg"]
        mock_isfile.return_value = True  # All files are considered files
        
        # Mock document processing - create different courses for each file
        mock_course1 = Mock()
        mock_course1.title = "Test Course 1"
        mock_chunks1 = [Mock(), Mock()]  # 2 chunks
        
        mock_course2 = Mock()
        mock_course2.title = "Test Course 2"
        mock_chunks2 = [Mock(), Mock()]  # 2 chunks
        
        mock_doc_proc.return_value.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),  # For course1.pdf
            (mock_course2, mock_chunks2)   # For course2.txt
            # ignore.jpg will be skipped due to file extension check
        ]
        mock_vector_store.return_value.get_existing_course_titles.return_value = []
        
        courses_added, chunks_added = rag_system.add_course_folder("/test/docs")
        
        # Should process 2 files (pdf and txt, not jpg)
        assert mock_doc_proc.return_value.process_course_document.call_count == 2
        assert courses_added == 2
        assert chunks_added == 4  # 2 courses × 2 chunks each