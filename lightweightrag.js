(function () {
    const initialState = window.__INITIAL_STATE__ || {};
    const presets = window.__PRESET_OPTIONS__ || {};

    const state = {
        chatbot: Array.isArray(initialState.chatbot) ? initialState.chatbot : [],
        logHistoryState: Array.isArray(initialState.log_history_state) ? initialState.log_history_state : [],
        busyMode: null,
    };

    const elements = {
        tabButtons: Array.from(document.querySelectorAll('.tab-button')),
        tabPanels: Array.from(document.querySelectorAll('.tab-panel')),
        chatMessages: document.getElementById('chatMessages'),
        questionInput: document.getElementById('questionInput'),
        sendButton: document.getElementById('sendButton'),
        clearButton: document.getElementById('clearButton'),
        clearConfirmRow: document.getElementById('clearConfirmRow'),
        confirmClearButton: document.getElementById('confirmClearButton'),
        cancelClearButton: document.getElementById('cancelClearButton'),
        presetSelect: document.getElementById('presetSelect'),
        topKRetInput: document.getElementById('topKRetInput'),
        topKRetValue: document.getElementById('topKRetValue'),
        topKCompInput: document.getElementById('topKCompInput'),
        topKCompValue: document.getElementById('topKCompValue'),
        thresholdInput: document.getElementById('thresholdInput'),
        thresholdValue: document.getElementById('thresholdValue'),
        multiTurnInput: document.getElementById('multiTurnInput'),
        currentProcessLog: document.getElementById('currentProcessLog'),
        logHtmlDisplay: document.getElementById('logHtmlDisplay'),
        retrievalResultsHtml: document.getElementById('retrievalResultsHtml'),
        citationPreviewHtml: document.getElementById('citationPreviewHtml'),
        onlineEvalHtml: document.getElementById('onlineEvalHtml'),
        knowledgeBaseStatusHtml: document.getElementById('knowledgeBaseStatusHtml'),
        buildReportHtml: document.getElementById('buildReportHtml'),
        dirInput: document.getElementById('dirInput'),
        chunkSizeInput: document.getElementById('chunkSizeInput'),
        overlapInput: document.getElementById('overlapInput'),
        buildButton: document.getElementById('buildButton'),
        refreshStatusButton: document.getElementById('refreshStatusButton'),
        buildStatus: document.getElementById('buildStatus'),
        buildLog: document.getElementById('buildLog'),
    };

    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function textToHtml(text) {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    function setPanelHtml(element, value) {
        if (!element) {
            return;
        }
        element.innerHTML = value || '';
    }

    function setInputValue(element, value) {
        if (!element) {
            return;
        }
        element.value = value == null ? '' : String(value);
    }

    function renderChatMessages(messages) {
        if (!elements.chatMessages) {
            return;
        }

        if (!Array.isArray(messages) || messages.length === 0) {
            elements.chatMessages.innerHTML = '<div class="chat-empty">当前还没有对话内容。</div>';
            return;
        }

        elements.chatMessages.innerHTML = messages
            .map((item, index) => {
                const role = item && item.role === 'user' ? 'user' : 'assistant';
                const roleLabel = role === 'user' ? '用户' : '助手';
                return [
                    `<div class="chat-item ${role}">`,
                    '<div class="chat-head">',
                    `<div class="chat-role">${roleLabel}</div>`,
                    `<button class="message-copy" type="button" data-copy-index="${index}">复制</button>`,
                    '</div>',
                    `<div class="chat-bubble">${textToHtml(item && item.content ? item.content : '')}</div>`,
                    '</div>',
                ].join('');
            })
            .join('');

        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }

    function updateRangeLabels() {
        if (!elements.topKRetInput || !elements.topKCompInput || !elements.thresholdInput) {
            return;
        }
        if (elements.topKRetValue) {
            elements.topKRetValue.textContent = elements.topKRetInput.value;
        }
        if (elements.topKCompValue) {
            elements.topKCompValue.textContent = elements.topKCompInput.value;
        }
        if (elements.thresholdValue) {
            elements.thresholdValue.textContent = Number(elements.thresholdInput.value).toFixed(2);
        }
    }

    function setBusy(mode) {
        const isBusy = Boolean(mode);
        const placeholder = mode === 'chat'
            ? '回答生成中，请稍候...'
            : mode === 'build'
                ? '知识库构建中，请稍候...'
                : '输入你的问题...';

        state.busyMode = mode;

        if (elements.questionInput) {
            elements.questionInput.disabled = isBusy;
            elements.questionInput.placeholder = placeholder;
        }
        if (elements.sendButton) {
            elements.sendButton.disabled = isBusy;
            elements.sendButton.textContent = mode === 'chat' ? '生成中...' : '发送';
        }
        if (elements.clearButton) {
            elements.clearButton.disabled = isBusy;
        }
        if (elements.confirmClearButton) {
            elements.confirmClearButton.disabled = isBusy;
        }
        if (elements.cancelClearButton) {
            elements.cancelClearButton.disabled = isBusy;
        }
        if (elements.buildButton) {
            elements.buildButton.disabled = isBusy;
            elements.buildButton.textContent = mode === 'build' ? '构建中...' : '构建索引';
        }
        if (elements.refreshStatusButton) {
            elements.refreshStatusButton.disabled = isBusy;
        }
    }

    function switchTab(target) {
        elements.tabButtons.forEach((button) => {
            button.classList.toggle('is-active', button.dataset.tabTarget === target);
        });
        elements.tabPanels.forEach((panel) => {
            panel.classList.toggle('is-active', panel.dataset.tabPanel === target);
        });
    }

    function applyChatPayload(payload) {
        state.chatbot = Array.isArray(payload.chatbot) ? payload.chatbot : [];
        state.logHistoryState = Array.isArray(payload.log_history_state) ? payload.log_history_state : [];

        renderChatMessages(state.chatbot);
        setInputValue(elements.currentProcessLog, payload.current_process_log);
        setPanelHtml(elements.logHtmlDisplay, payload.log_html_display);
        setPanelHtml(elements.retrievalResultsHtml, payload.retrieval_results_html);
        setPanelHtml(elements.citationPreviewHtml, payload.citation_preview_html);
        setPanelHtml(elements.onlineEvalHtml, payload.online_eval_html);
    }

    function applyClearPayload(payload) {
        state.chatbot = Array.isArray(payload.chatbot) ? payload.chatbot : [];
        state.logHistoryState = Array.isArray(payload.log_history_state) ? payload.log_history_state : [];

        renderChatMessages(state.chatbot);
        setInputValue(elements.currentProcessLog, payload.current_process_log);
        setPanelHtml(elements.logHtmlDisplay, payload.log_html_display);
        setPanelHtml(elements.retrievalResultsHtml, payload.retrieval_results_html);
        setPanelHtml(elements.citationPreviewHtml, payload.citation_preview_html);
        setPanelHtml(elements.onlineEvalHtml, payload.online_eval_html);

        if (elements.clearConfirmRow) {
            elements.clearConfirmRow.classList.add('is-hidden');
        }
    }

    function applyKnowledgeBasePanels(payload) {
        setPanelHtml(elements.knowledgeBaseStatusHtml, payload.knowledge_base_status_html);
        setPanelHtml(elements.buildReportHtml, payload.build_report_html);
    }

    function applyBuildPayload(payload) {
        setInputValue(elements.buildStatus, payload.build_status);
        setInputValue(elements.buildLog, payload.build_log);
        applyKnowledgeBasePanels(payload);
    }
    async function fetchJson(url, options) {
        const requestOptions = Object.assign({ cache: 'no-store' }, options || {});
        const response = await fetch(url, requestOptions);
        const contentType = response.headers.get('content-type') || '';

        if (!response.ok) {
            if (contentType.includes('application/json')) {
                const payload = await response.json();
                throw new Error(payload.error || `请求失败：${response.status}`);
            }
            throw new Error(`请求失败：${response.status}`);
        }

        return response.json();
    }

    async function handleChatSubmit() {
        if (state.busyMode) {
            return;
        }
        if (!elements.questionInput) {
            return;
        }

        const question = elements.questionInput.value;
        if (!String(question || '').trim()) {
            return;
        }

        setBusy('chat');
        elements.questionInput.value = '';

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question,
                    history: state.chatbot,
                    top_k_ret: Number(elements.topKRetInput.value),
                    top_k_comp: Number(elements.topKCompInput.value),
                    threshold: Number(elements.thresholdInput.value),
                    multi_turn_enabled: elements.multiTurnInput.checked,
                    log_history_state: state.logHistoryState,
                }),
            });

            if (!response.ok || !response.body) {
                throw new Error(`请求失败：${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

                let newlineIndex = buffer.indexOf('\n');
                while (newlineIndex >= 0) {
                    const line = buffer.slice(0, newlineIndex).trim();
                    buffer = buffer.slice(newlineIndex + 1);
                    if (line) {
                        const message = JSON.parse(line);
                        if (message.type === 'update' && message.data) {
                            applyChatPayload(message.data);
                        } else if (message.type === 'error') {
                            throw new Error(message.error || '流式请求失败');
                        }
                    }
                    newlineIndex = buffer.indexOf('\n');
                }

                if (done) {
                    const lastLine = buffer.trim();
                    if (lastLine) {
                        const message = JSON.parse(lastLine);
                        if (message.type === 'update' && message.data) {
                            applyChatPayload(message.data);
                        } else if (message.type === 'error') {
                            throw new Error(message.error || '流式请求失败');
                        }
                    }
                    break;
                }
            }
        } catch (error) {
            alert(error.message || '发送失败');
        } finally {
            setBusy(null);
            if (elements.questionInput) {
                elements.questionInput.focus();
            }
        }
    }

    async function handleConversationClear() {
        if (state.busyMode) {
            return;
        }

        setBusy('chat');
        try {
            const payload = await fetchJson('/api/conversation/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            applyClearPayload(payload);
        } catch (error) {
            alert(error.message || '清空会话失败');
        } finally {
            setBusy(null);
        }
    }

    async function handleKnowledgeBasePanelsRefresh() {
        if (state.busyMode) {
            return;
        }

        setBusy('build');
        try {
            const payload = await fetchJson('/api/knowledge-base/panels');
            applyKnowledgeBasePanels(payload);
        } catch (error) {
            alert(error.message || '刷新状态失败');
        } finally {
            setBusy(null);
        }
    }

    async function handleKnowledgeBaseBuild() {
        if (state.busyMode) {
            return;
        }

        setBusy('build');
        setInputValue(elements.buildStatus, '构建中...');
        setInputValue(elements.buildLog, '正在执行，请稍候...');

        try {
            const payload = await fetchJson('/api/knowledge-base/build', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source_dir: elements.dirInput ? elements.dirInput.value : '',
                    chunk_size: elements.chunkSizeInput ? elements.chunkSizeInput.value : '',
                    overlap: elements.overlapInput ? elements.overlapInput.value : '',
                }),
            });
            applyBuildPayload(payload);
        } catch (error) {
            setInputValue(elements.buildStatus, '失败');
            setInputValue(elements.buildLog, error.message || '构建失败');
        } finally {
            setBusy(null);
        }
    }
    function applyPresetByName(name) {
        const preset = presets[name];
        if (!preset || !elements.topKRetInput || !elements.topKCompInput || !elements.thresholdInput) {
            return;
        }
        elements.topKRetInput.value = preset.top_k_ret;
        elements.topKCompInput.value = preset.top_k_comp;
        elements.thresholdInput.value = preset.threshold;
        updateRangeLabels();
    }

    function bindEvents() {
        elements.tabButtons.forEach((button) => {
            button.addEventListener('click', () => switchTab(button.dataset.tabTarget));
        });

        if (elements.presetSelect) {
            elements.presetSelect.addEventListener('change', () => {
                applyPresetByName(elements.presetSelect.value);
            });
        }

        if (elements.topKRetInput) {
            elements.topKRetInput.addEventListener('input', updateRangeLabels);
        }
        if (elements.topKCompInput) {
            elements.topKCompInput.addEventListener('input', updateRangeLabels);
        }
        if (elements.thresholdInput) {
            elements.thresholdInput.addEventListener('input', updateRangeLabels);
        }

        if (elements.sendButton) {
            elements.sendButton.addEventListener('click', handleChatSubmit);
        }
        if (elements.questionInput) {
            elements.questionInput.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    handleChatSubmit();
                }
            });
        }

        if (elements.clearButton) {
            elements.clearButton.addEventListener('click', () => {
                if (state.busyMode || !elements.clearConfirmRow) {
                    return;
                }
                elements.clearConfirmRow.classList.remove('is-hidden');
            });
        }

        if (elements.chatMessages) {
            elements.chatMessages.addEventListener('click', async (event) => {
                const copyButton = event.target.closest('[data-copy-index]');
                if (!copyButton) {
                    return;
                }

                const messageIndex = Number(copyButton.dataset.copyIndex);
                const message = state.chatbot[messageIndex];
                if (!message) {
                    return;
                }

                try {
                    await navigator.clipboard.writeText(message.content || '');
                    const previousLabel = copyButton.textContent;
                    copyButton.textContent = '已复制';
                    window.setTimeout(() => {
                        copyButton.textContent = previousLabel;
                    }, 1200);
                } catch (error) {
                    alert('复制失败');
                }
            });
        }

        if (elements.cancelClearButton) {
            elements.cancelClearButton.addEventListener('click', () => {
                if (elements.clearConfirmRow) {
                    elements.clearConfirmRow.classList.add('is-hidden');
                }
            });
        }
        if (elements.confirmClearButton) {
            elements.confirmClearButton.addEventListener('click', handleConversationClear);
        }

        if (elements.refreshStatusButton) {
            elements.refreshStatusButton.addEventListener('click', handleKnowledgeBasePanelsRefresh);
        }
        if (elements.buildButton) {
            elements.buildButton.addEventListener('click', handleKnowledgeBaseBuild);
        }
    }

    function init() {
        renderChatMessages(state.chatbot);
        updateRangeLabels();
        bindEvents();
    }

    init();
}());



