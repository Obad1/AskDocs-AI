/**
 * AskDocs AI — Client-side AI Engine using Transformers.js
 * All models run in the browser. Downloads once, cached in IndexedDB.
 */

const AI = {
    models: {},
    loaded: false,
    loading: false,
    progress: { current: '', percent: 0, message: '' },
    callbacks: [],

    // Model registry — all from Xenova (pre-converted ONNX)
    MODELS: {
        summarizer: 'Xenova/distilbart-cnn-6-6',
        ner: 'Xenova/distilbert-base-uncased-finetuned-ner',
        qa: 'Xenova/flan-t5-small',
    },

    onProgress(cb) {
        this.callbacks.push(cb);
    },

    _notify() {
        this.callbacks.forEach(cb => cb({ ...this.progress }));
    },

    async load() {
        if (this.loaded || this.loading) return;
        this.loading = true;

        const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1');
        
        // Allow loading from any origin (for CDN models)
        env.allowLocalModels = false;

        // 1. Load NER model
        this.progress = { current: 'ner', percent: 10, message: 'Loading entity extractor...' };
        this._notify();
        try {
            this.models.ner = await pipeline('token-classification', this.MODELS.ner, {
                progress_callback: (p) => {
                    if (p.status === 'progress') {
                        this.progress = { current: 'ner', percent: Math.round(p.progress * 0.25), message: `Downloading NER model... ${p.progress}%` };
                        this._notify();
                    }
                }
            });
        } catch (e) {
            console.warn('NER model failed:', e);
        }

        // 2. Load summarizer
        this.progress = { current: 'summarizer', percent: 35, message: 'Loading summarizer...' };
        this._notify();
        try {
            this.models.summarizer = await pipeline('summarization', this.MODELS.summarizer, {
                progress_callback: (p) => {
                    if (p.status === 'progress') {
                        this.progress = { current: 'summarizer', percent: 35 + Math.round(p.progress * 0.3), message: `Downloading summarizer... ${p.progress}%` };
                        this._notify();
                    }
                }
            });
        } catch (e) {
            console.warn('Summarizer model failed:', e);
        }

        // 3. Load QA/generation model
        this.progress = { current: 'qa', percent: 70, message: 'Loading Q&A engine...' };
        this._notify();
        try {
            this.models.qa = await pipeline('text2text-generation', this.MODELS.qa, {
                progress_callback: (p) => {
                    if (p.status === 'progress') {
                        this.progress = { current: 'qa', percent: 70 + Math.round(p.progress * 0.25), message: `Downloading Q&A model... ${p.progress}%` };
                        this._notify();
                    }
                }
            });
        } catch (e) {
            console.warn('Q&A model failed:', e);
        }

        this.progress = { current: 'done', percent: 100, message: 'All models ready!' };
        this._notify();
        this.loaded = true;
        this.loading = false;
    },

    // --- Named Entity Recognition ---
    async extractEntities(text) {
        if (!this.models.ner) return this._regexNER(text);
        
        const entities = [];
        const chunks = this._splitChunks(text, 512);
        
        for (const chunk of chunks) {
            try {
                const result = await this.models.ner(chunk);
                let current = { text: '', type: '' };
                for (const item of result) {
                    const label = item.entity.replace('##', '');
                    const type = item.entity.startsWith('B-') ? item.entity.slice(2) : 
                                 item.entity.startsWith('I-') ? item.entity.slice(2) : '';
                    
                    if (item.entity.startsWith('B-')) {
                        if (current.text) entities.push(current);
                        current = { text: label, type };
                    } else if (item.entity.startsWith('I-') && current.text) {
                        current.text += label.replace('##', '');
                    } else {
                        if (current.text) entities.push(current);
                        current = { text: '', type: '' };
                    }
                }
                if (current.text) entities.push(current);
            } catch (e) {
                console.warn('NER failed on chunk:', e);
            }
        }
        
        return entities.filter(e => e.text.length > 2);
    },

    _regexNER(text) {
        const entities = [];
        const matches = text.matchAll(/\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b/g);
        for (const m of matches) {
            entities.push({ text: m[1], type: 'TERM' });
        }
        return entities;
    },

    // --- Summarization ---
    async summarize(text, detail = 'standard') {
        if (this.models.summarizer) {
            try {
                const chunks = this._splitChunks(text, 1024);
                let summaries = [];
                
                for (const chunk of chunks.slice(0, 3)) {
                    const maxLen = detail === 'brief' ? 80 : detail === 'detailed' ? 200 : 130;
                    const result = await this.models.summarizer(chunk, {
                        max_length: maxLen,
                        min_length: 30,
                    });
                    summaries.push(result[0].summary_text);
                }
                
                return summaries.join(' ');
            } catch (e) {
                console.warn('Summarization failed:', e);
            }
        }
        
        // Fallback: extractive
        return this._extractiveSummary(text, detail);
    },

    _extractiveSummary(text, detail) {
        const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 20);
        const n = detail === 'brief' ? 3 : detail === 'detailed' ? 8 : 5;
        
        // Score sentences by position and content
        const scored = sentences.map((s, i) => {
            let score = 0;
            if (i === 0) score += 5;
            else if (i === 1) score += 3;
            const words = s.split(/\s+/);
            if (words.length > 10 && words.length < 40) score += 2;
            if (/\b(is|are|key|important|main|fundamental)\b/i.test(s)) score += 1;
            return { score, index: i, sentence: s };
        });
        
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, n).sort((a, b) => a.index - b.index);
        return top.map(s => s.sentence).join(' ');
    },

    // --- Question Generation (using T5 prompts) ---
    async generateQuestions(context, answers, count = 5) {
        const questions = [];
        
        for (const answer of answers.slice(0, count)) {
            if (answer.length < 3 || answer.length > 50) continue;
            
            // Find the sentence containing this answer
            const sentences = context.split(/(?<=[.!?])\s+/);
            const sent = sentences.find(s => s.toLowerCase().includes(answer.toLowerCase()));
            const sentenceContext = sent || context;
            
            if (this.models.qa) {
                try {
                    const prompt = `generate question: ${sentenceContext.replace(new RegExp(this._escapeRegex(answer), 'i'), '[answer]')}`;
                    const result = await this.models.qa(prompt, {
                        max_length: 64,
                        num_beams: 2,
                    });
                    let question = result[0].generated_text.trim();
                    
                    if (question && question.length > 5) {
                        if (!question.endsWith('?')) question += '?';
                        questions.push({ question, answer, type: 'qa' });
                    }
                } catch (e) {
                    console.warn('QG failed:', e);
                }
            }
            
            // Fallback: template-based
            if (questions.length === 0) {
                const templateQ = this._templateQuestion(context, answer);
                if (templateQ) {
                    questions.push({ question: templateQ, answer, type: 'template' });
                }
            }
        }
        
        return questions;
    },

    _templateQuestion(context, answer) {
        const lower = context.toLowerCase();
        const idx = lower.indexOf(answer.toLowerCase());
        if (idx === -1) return null;
        
        // Look for definition patterns
        const before = context.substring(0, idx);
        const after = context.substring(idx + answer.length);
        
        if (/\b(is|are|refers to|means|defined as)\b/i.test(before + after)) {
            return `What is ${answer}?`;
        }
        if (/\b(enables|allows|provides|uses)\b/i.test(after)) {
            return `What does ${answer} do?`;
        }
        if (/\b(because|therefore|thus|leads to)\b/i.test(before + after)) {
            return `Why is ${answer} important?`;
        }
        
        return null;
    },

    // --- Q&A ---
    async answerQuestion(question, context) {
        if (this.models.qa) {
            try {
                const prompt = `Answer the question based on the following text:\n\n${context.substring(0, 800)}\n\nQuestion: ${question}`;
                const result = await this.models.qa(prompt, {
                    max_length: 128,
                    num_beams: 2,
                });
                return result[0].generated_text.trim();
            } catch (e) {
                console.warn('QA failed:', e);
            }
        }
        
        // Fallback: search-based
        return this._searchAnswer(question, context);
    },

    _searchAnswer(question, context) {
        const qWords = new Set(question.toLowerCase().split(/\s+/).filter(w => w.length > 3));
        const sentences = context.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 20);
        
        const scored = sentences.map(s => {
            const sWords = new Set(s.toLowerCase().split(/\s+/));
            let score = 0;
            for (const w of qWords) {
                if (sWords.has(w)) score += 1;
            }
            return { score, sentence: s };
        });
        
        scored.sort((a, b) => b.score - a.score);
        return scored[0]?.sentence || 'I could not find an answer in this document.';
    },

    // --- Flashcard Generation ---
    async generateFlashcards(text, count = 10) {
        const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 30 && s.trim().length < 300);
        const entities = await this.extractEntities(text);

        const cards = [];

        // Find good answer candidates from entities
        const answerCandidates = entities
            .filter(e => e.text.length >= 3 && e.text.length <= 50)
            .map(e => e.text);

        // For each answer, find the sentence containing it and generate a question
        for (const answer of answerCandidates.slice(0, count * 2)) {
            if (cards.length >= count) break;

            // Find sentence containing this answer
            const sent = sentences.find(s => s.toLowerCase().includes(answer.toLowerCase()));
            if (!sent) continue;

            let question = null;

            // Use QG model to generate question
            if (this.models.qa) {
                try {
                    const prompt = `generate question: ${sent.replace(new RegExp(this._escapeRegex(answer), 'i'), '[answer]')}`;
                    const result = await this.models.qa(prompt, {
                        max_length: 64,
                        num_beams: 2,
                    });
                    question = result[0].generated_text.trim();
                    if (question && !question.endsWith('?')) question += '?';
                } catch (e) {
                    console.warn('QG failed:', e);
                }
            }

            // Fallback: template-based question
            if (!question) {
                question = this._makeFlashcardQuestion(sent, answer);
            }

            if (question) {
                cards.push({
                    question,
                    answer,
                    type: 'qa',
                    difficulty: this._estimateDifficulty(question, answer),
                });
            }
        }

        // Add fill-in-blank cards if needed
        if (cards.length < count) {
            for (const sent of sentences) {
                if (cards.length >= count) break;
                const blankCard = this._makeBlankCard(sent, entities);
                if (blankCard) cards.push(blankCard);
            }
        }

        return cards.slice(0, count);
    },

    _makeFlashcardQuestion(sentence, entity) {
        const lower = sentence.toLowerCase();
        const entityLower = entity.toLowerCase();
        
        if (/\b(is|are|refers to|means|defined as)\b/i.test(sentence)) {
            return `What is ${entity}?`;
        }
        if (/\b(enables|allows|provides|uses|facilitates)\b/i.test(sentence)) {
            return `What role does ${entity} play?`;
        }
        if (/\b(because|therefore|thus|leads to|causes)\b/i.test(sentence)) {
            return `Why is ${entity} important?`;
        }
        
        return `What can you tell about ${entity}?`;
    },

    _makeBlankCard(sentence, entities) {
        // Pick the longest entity as the blank
        const valid = entities.filter(e => e.text.length > 3 && e.text.length < 40);
        if (valid.length === 0) return null;
        
        valid.sort((a, b) => b.text.length - a.text.length);
        const term = valid[0].text;
        
        if (!sentence.toLowerCase().includes(term.toLowerCase())) return null;
        
        const blanked = sentence.replace(new RegExp(this._escapeRegex(term), 'i'), '________');
        if (!blanked.includes('________')) return null;
        
        return {
            question: `Fill in the blank: ${blanked}`,
            answer: term,
            type: 'fill-blank',
            difficulty: 'medium',
        };
    },

    _estimateDifficulty(question, answer) {
        const score = question.split(/\s+/).length + answer.split(/\s+/).length;
        if (score > 15) return 'hard';
        if (score > 8) return 'medium';
        return 'easy';
    },

    // --- Quiz Generation ---
    async generateQuiz(text, count = 10) {
        const entities = await this.extractEntities(text);
        const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 30);
        const concepts = this._extractConcepts(sentences);
        
        const mcQuestions = [];
        const tfQuestions = [];
        const fbQuestions = [];
        
        // Multiple choice from concepts
        const allTerms = concepts.map(c => c.term).filter(t => t.length > 3);
        const allDefs = concepts.map(c => c.definition).filter(d => d.length > 10);
        
        for (const concept of concepts) {
            if (mcQuestions.length >= Math.ceil(count * 0.5)) break;
            
            const correct = concept.definition;
            const distractors = allDefs.filter(d => d !== correct).slice(0, 3);
            while (distractors.length < 3) distractors.push('None of the above');
            
            const options = [correct, ...distractors.slice(0, 3)];
            this._shuffle(options);
            
            mcQuestions.push({
                question: `What is ${concept.term}?`,
                options,
                correct: options.indexOf(correct),
                explanation: `${concept.term}: ${concept.definition}`,
                points: 2,
            });
        }
        
        // True/False
        for (const concept of concepts) {
            if (tfQuestions.length >= Math.ceil(count * 0.25)) break;
            
            // True statement
            tfQuestions.push({
                question: `${concept.term} ${concept.definition}`,
                correct: true,
                explanation: `This is correct based on the document.`,
                points: 1,
            });
            
            // False statement (swap definition)
            const others = concepts.filter(c => c.term !== concept.term);
            if (others.length > 0) {
                const falseDef = others[Math.floor(Math.random() * others.length)].definition;
                tfQuestions.push({
                    question: `${concept.term} ${falseDef}`,
                    correct: false,
                    explanation: `False. ${concept.term} actually ${concept.definition}`,
                    points: 1,
                });
            }
        }
        
        // Fill in blank
        for (const sent of sentences) {
            if (fbQuestions.length >= Math.ceil(count * 0.25)) break;
            
            const sentEntities = entities.filter(e => sent.toLowerCase().includes(e.text.toLowerCase()));
            for (const ent of sentEntities) {
                if (ent.text.length < 3 || ent.text.length > 30) continue;
                
                const blanked = sent.replace(new RegExp(this._escapeRegex(ent.text), 'i'), '______');
                if (!blanked.includes('______')) continue;
                
                fbQuestions.push({
                    question: `Fill in the blank: ${blanked}`,
                    answer: ent.text,
                    hint: sent.substring(0, 80) + '...',
                    points: 1,
                });
                break;
            }
        }
        
        this._shuffle(tfQuestions);
        
        return {
            multiple_choice: mcQuestions,
            true_false: tfQuestions,
            fill_blank: fbQuestions,
        };
    },

    _extractConcepts(sentences) {
        const concepts = [];
        const seen = new Set();
        
        for (const sent of sentences) {
            const patterns = [
                /([A-Z][A-Za-z\s]{2,40})\s+(?:is|are|was|were|refers to|means|defines|describes)\s+([^.,!?]+)/,
                /([A-Z][A-Za-z\s]{2,30})\s+(?:enables|allows|provides|supports|facilitates)\s+([^.,!?]+)/,
            ];
            
            for (const pattern of patterns) {
                const match = sent.match(pattern);
                if (match) {
                    const term = match[1].trim();
                    const definition = match[2].trim();
                    if (!seen.has(term.toLowerCase()) && definition.length > 10) {
                        seen.add(term.toLowerCase());
                        concepts.push({ term, definition, sentence: sent });
                    }
                }
            }
        }
        
        return concepts;
    },

    _escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    },

    _shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    },

    _splitChunks(text, maxLen) {
        const sentences = text.split(/(?<=[.!?])\s+/);
        const chunks = [];
        let current = '';
        
        for (const sent of sentences) {
            if ((current + sent).length > maxLen && current) {
                chunks.push(current.trim());
                current = '';
            }
            current += sent + ' ';
        }
        if (current.trim()) chunks.push(current.trim());
        return chunks;
    },
};
