let mapping = {};


const indexCache = {
    data: new Map(),
    preloadQueue: new Set(),
    preloadPromises: new Map(),
    
    async get(groupIndex, baseDir) {
        const cacheKey = `${baseDir}/${groupIndex}`;
        if (this.data.has(cacheKey)) {
            return this.data.get(cacheKey);
        }

        if(this.preloadPromises.has(cacheKey)) {
            return this.preloadPromises.get(cacheKey);
        }

        try {
            const indexData = await this._fetchIndex(groupIndex, baseDir);
            return indexData;
        } catch (error) {
            throw error;
        }
    },

    async preload(groupIndex, baseDir) {
        const cacheKey = `${baseDir}/${groupIndex}`;
        if (this.data.has(cacheKey) || this.preloadQueue.has(cacheKey)) {
            return;
        }

        this.preloadQueue.add(cacheKey);
        
        const promise = this._fetchIndex(groupIndex, baseDir);
        this.preloadPromises.set(cacheKey, promise);

        try {
            await promise;
        } catch (error) {
            console.error(`[IndexCache] Failed to preload ${groupIndex}.index:`, error);
        } finally {
            this.preloadQueue.delete(cacheKey);
            this.preloadPromises.delete(cacheKey);
        }
    },

    async _fetchIndex(groupIndex, baseDir) {
        const cacheKey = `${baseDir}/${groupIndex}`;
        const indexResponse = await fetch(`${baseDir}/${groupIndex}.index`);
        if (!indexResponse.ok) {
            throw new Error(`${indexResponse.status} ${indexResponse.statusText}`);
        }
        
        const compressedData = await indexResponse.arrayBuffer();
        
        const ds = new DecompressionStream('gzip');
        const decompressedStream = new Response(compressedData).body.pipeThrough(ds);
        const decompressedData = await new Response(decompressedStream).arrayBuffer();

        this.data.set(cacheKey, decompressedData);
        return decompressedData;
    }
};

const watermarkImage = new Image();
watermarkImage.src = 'watermark.png';
let watermarkLoaded = false;
watermarkImage.onload = () => {
    watermarkLoaded = true;
};

async function extractFrame(folderId, frameNum, baseDir = '') {
    const groupIndex = Math.floor((folderId - 1) / 10);
    
    try {
        const indexData = await indexCache.get(groupIndex, baseDir);
        const dataView = new DataView(indexData);
        let offset = 0;
        
        const gridW = dataView.getUint32(offset, true); offset += 4;
        const gridH = dataView.getUint32(offset, true); offset += 4;
        
        const folderCount = dataView.getUint32(offset, true); offset += 4;
        offset += folderCount * 4;
        const fileCount = dataView.getUint32(offset, true); offset += 4;
    
        let left = 0;
        let right = fileCount - 1;
        let startOffset = null;
        let endOffset = null;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const recordOffset = offset + mid * 16;
            
            const currFolder = dataView.getUint32(recordOffset, true);
            const currFrame = dataView.getUint32(recordOffset + 4, true);
            const currFileOffset = Number(dataView.getBigUint64(recordOffset + 8, true));
            
            if (currFolder === folderId && currFrame === frameNum) {
                startOffset = currFileOffset;
                if (mid < fileCount - 1) {
                    endOffset = Number(dataView.getBigUint64(recordOffset + 24, true));
                }
                break;
            } else if (currFolder < folderId || (currFolder === folderId && currFrame < frameNum)) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        if (startOffset === null) {
            throw new Error();
        }
        
        const response = await fetch(`${baseDir}/${groupIndex}.webp`, {
            headers: {
                'Range': `bytes=${startOffset}-${endOffset ? endOffset - 1 : ''}`
            }
        });

        if (!response.ok) {
            throw new Error(`${response.status} ${response.statusText}`);
        }

        const data = await response.blob();
        return new Blob([data], {type: 'image/webp'});
        
    } catch (error) {
        console.error(`[Frame] Failed to extract frame P${folderId}/${frameNum}:`, error);
        throw error;
    }
} 
async function loadMapping() {
    try {
        const response = await fetch('./mapping.json');
        if (!response.ok) throw new Error("Failed to load mapping.json");
        return await response.json();
    } catch (error) {
        console.error('[Mapping] Failed to load:', error);
        return {};
    }
}

const AppState = {
    isSearching: false,
    randomStringDisplayed: false,
    searchResults: [],
    currentPage: 1,
    itemsPerPage: 20,
    hasMoreResults: true,
    cachedResults: [],
    displayedCount: 0,
    showWatermark: true
};


const CONFIG = {
    randomStrings: ["探索VV的开源世界", "为东大助力", "搜索你想要的内容"],
    apiBaseUrl: ''
};


class UIController {
    static updateSearchFormPosition(isSearching) {
        const searchForm = document.getElementById('searchForm');
        const randomStringDisplay = document.getElementById('randomStringDisplay');
        
        if (isSearching) {
            searchForm.classList.add('searching');
            if (!AppState.randomStringDisplayed) {
                this.showRandomString();
            }
        } else {
            searchForm.classList.remove('searching');
            if (AppState.cachedResults.length > 0) {
                this.clearRandomString();
            }
        }
    }

    static showRandomString() {
        if (!AppState.randomStringDisplayed) {
            const randomStringDisplay = document.getElementById('randomStringDisplay');
            const randomIndex = Math.floor(Math.random() * CONFIG.randomStrings.length);
            randomStringDisplay.textContent = CONFIG.randomStrings[randomIndex];
            AppState.randomStringDisplayed = true;
            
            randomStringDisplay.classList.remove('fade-out');
            randomStringDisplay.classList.add('fade-in');
        }
    }

    static clearRandomString() {
        const randomStringDisplay = document.getElementById('randomStringDisplay');
        randomStringDisplay.classList.remove('fade-in');
        randomStringDisplay.classList.add('fade-out');
        
        setTimeout(() => {
            randomStringDisplay.textContent = '';
            AppState.randomStringDisplayed = false;
        }, 300);
    }
}

class SearchController {
    static async performSearch(query, minRatio, minSimilarity) {
        const url = `${CONFIG.apiBaseUrl}/search?query=${encodeURIComponent(query)}&min_ratio=${minRatio}&min_similarity=${minSimilarity}`;
        
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error("Network request failed");
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let totalBytes = 0;
            
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, {stream: true});
                totalBytes += value.length;
                
                const progress = Math.min(95, (totalBytes / response.headers.get('Content-Length')) * 100 * 1.2);
                document.getElementById('loadingBar').style.width = `${progress}%`;
                
                let lines = buffer.split('\n');
                buffer = lines.pop();
                
                for (let line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const result = JSON.parse(line);
                        if (result) {
                            AppState.cachedResults.push(result);
                        }
                    } catch (e) {
                        console.error('[Search] Failed to parse single result:', e);
                    }
                }
            }
            
            if (buffer.trim()) {
                try {
                    const result = JSON.parse(buffer);
                    if (result) {
                        AppState.cachedResults.push(result);
                    }
                } catch (e) {
                    console.error('[Search] Failed to parse final result:', e);
                }
            }
            
            if (AppState.cachedResults.length > 0) {
                displayResults({
                    status: 'success',
                    data: AppState.cachedResults,
                    count: AppState.cachedResults.length
                }, false);
                
                return {
                    status: 'success',
                    data: AppState.cachedResults,
                    count: AppState.cachedResults.length
                };
            } else {
                return {
                    status: 'success',
                    data: [],
                    count: 0
                };
            }
        } catch (error) {
            console.error('[Search] Failed:', error);
            throw error;
        } finally {
            completeLoadingBar();
        }
    }

    static validateSearchInput(query) {
        return query.trim() !== "";
    }
}

async function handleSearch(mapping) {
    const query = document.getElementById('query').value.trim();
    const minRatio = document.getElementById('minRatio').value;
    const minSimilarity = document.getElementById('minSimilarity').value;

    if (!SearchController.validateSearchInput(query)) {
        alert("请输入搜索关键词！");
        return;
    }

    try {
        startLoadingBar();
        UIController.showRandomString();
        UIController.updateSearchFormPosition(true);
        document.getElementById('results').innerHTML = '';
        
        AppState.currentPage = 1;
        AppState.cachedResults = [];
        AppState.displayedCount = 0;
        AppState.hasMoreResults = true;

        await SearchController.performSearch(query, minRatio, minSimilarity);
        
        initializeScrollListener();

    } catch (error) {
        console.error('[Search] Failed:', error);
        document.getElementById('results').innerHTML = '<div class="result-card">搜索失败，请稍后重试</div>';
    } finally {
        UIController.updateSearchFormPosition(false);
    }
}

async function initializeApp() {
    try {
        mapping = await loadMapping();
        initializeScrollListener();

        for (let i = 0; i <= 26; i++) {
            indexCache.preload(i, 'https://vv.noxylva.org').catch(error => {
                console.error(`[IndexCache] Failed to preload ${i}.index:`, error);
            });
        }
        
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            if (AppState.isSearching) return;
            
            AppState.isSearching = true;
            try {
                await handleSearch(mapping);
            } finally {
                AppState.isSearching = false;
            }
        });

       
        document.getElementById('query').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (AppState.isSearching) return;
                document.getElementById('searchForm').dispatchEvent(new Event('submit'));
            }
        });

        document.getElementById('refreshDiv').addEventListener('click', function() {
            location.reload();
        });

        
    } catch (error) {
        console.error('[App] Initialization failed:', error);
    }
}


document.addEventListener('DOMContentLoaded', () => {
    initializeApp();

    const toggleButton = document.getElementById('toggleAdvancedOptions');
    const advancedOptions = document.getElementById('advancedOptions');
    
    toggleButton.addEventListener('click', () => {
        const isExpanded = advancedOptions.classList.contains('show');
        
        if (!isExpanded) {

            advancedOptions.style.transition = 'none';
            advancedOptions.classList.add('show');
            const height = advancedOptions.scrollHeight;
            advancedOptions.classList.remove('show');
            
            void advancedOptions.offsetHeight;
            advancedOptions.style.transition = '';
            advancedOptions.style.maxHeight = height + 'px';
            advancedOptions.classList.add('show');
        } else {
            advancedOptions.style.maxHeight = '0';
            advancedOptions.classList.remove('show');
        }
        
        toggleButton.classList.toggle('active');
        toggleButton.setAttribute('aria-expanded', !isExpanded);
    });

    const watermarkToggle = document.getElementById('watermarkToggle');
    watermarkToggle.addEventListener('change', () => {
        AppState.showWatermark = watermarkToggle.checked;
        
        if (window.canvasRenderQueue) {
            window.canvasRenderQueue.forEach(canvas => {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(canvas.originalCanvas, 0, 0);
                
                if (AppState.showWatermark && watermarkLoaded) {
                    const watermarkScale = canvas.width * 0.25 / watermarkImage.width;
                    const watermarkWidth = watermarkImage.width * watermarkScale;
                    const watermarkHeight = watermarkImage.height * watermarkScale;
                    
                    ctx.drawImage(
                        watermarkImage,
                        canvas.width - watermarkWidth - 5,
                        canvas.height - watermarkHeight - 5,
                        watermarkWidth,
                        watermarkHeight
                    );
                }
            });
        }
    });
});

function displayResults(data, append = false) {
    const resultsDiv = document.getElementById('results');
    
    if (!append) {
        resultsDiv.innerHTML = '';
        AppState.displayedCount = 0;
    }
    
    if (!data.data || data.data.length === 0 || data.data[0].count === 0) {
        if (!append) {
            const query = document.getElementById('query').value.trim();
            const minRatio = parseFloat(document.getElementById('minRatio').value).toFixed(1);
            const minSimilarity = parseFloat(document.getElementById('minSimilarity').value).toFixed(1);
            
            resultsDiv.innerHTML = `
                <div class="error-message">
                    <h3>未找到与 "${query}" 匹配的结果</h3>
                    <p>建议：</p>
                    <ul>
                        <li>检查输入是否正确</li>
                        <li>尝试降低最小匹配率（当前：${minRatio}%）</li>
                        <li>尝试降低最小相似度（当前：${minSimilarity}）</li>
                        <li>尝试使用更简短的关键词</li>
                    </ul>
                </div>`;
        }
        AppState.hasMoreResults = false;
        return;
    }

    const fragment = document.createDocumentFragment();
    
    const startIndex = AppState.displayedCount;
    const endIndex = Math.min(startIndex + AppState.itemsPerPage, data.data.length);
    const newResults = data.data.slice(startIndex, endIndex);

    if (endIndex >= data.data.length) {
        AppState.hasMoreResults = false;
    }

    const cards = newResults.map(result => {
        if (!result || typeof result !== 'object') return null;
        
        const card = document.createElement('div');
        card.className = 'result-card';
        card.addEventListener('click', () => handleCardClick(result));
        card.style.cursor = 'pointer';
        
        const episodeMatch = result.filename ? result.filename.match(/\[P(\d+)\]/) : null;
        const timeMatch = result.timestamp ? result.timestamp.match(/^(\d+)m(\d+)s$/) : null;
        
        const cleanFilename = result.filename
            ? result.filename
                .replace(/\[P(\d+)\].*?\s+/, 'P$1 ')
                .replace(/\.json$/, '')
                .trim()
            : '';

        const cardContent = `
            <div class="result-content">
                <h3>${episodeMatch ? `<span class="tag">${episodeMatch[1]}</span>${cleanFilename.replace(/P\d+/, '').trim()}` : cleanFilename}</h3>
                <p class="result-text">${result.text || ''}</p>
                ${result.timestamp ? `
                <p class="result-meta">
                    ${result.timestamp} · 
                    匹配度 ${result.match_ratio ? parseFloat(result.match_ratio).toFixed(1) : 0}% · 
                    相似度 ${result.similarity ? (result.similarity * 100).toFixed(1) : 0}%
                </p>` : ''}
            </div>
        `;

        card.innerHTML = cardContent;
        return card;
    }).filter(Boolean);

    cards.forEach(card => fragment.appendChild(card));

    resultsDiv.appendChild(fragment);

    requestAnimationFrame(() => {
        cards.forEach((card, index) => {
            const result = newResults[index];
            loadPreviewImage(card, result);
        });
    });

    AppState.displayedCount = endIndex;
    AppState.hasMoreResults = endIndex < data.data.length;

    const trigger = document.getElementById('scroll-trigger');
    if (trigger) {
        resultsDiv.appendChild(trigger);
    }
}


async function loadPreviewImage(card, result) {
    const episodeMatch = result.filename?.match(/\[P(\d+)\]/);
    const timeMatch = result.timestamp?.match(/^(\d+)m(\d+)s$/);
    
    if (!episodeMatch || !timeMatch) return;
    
    const episodeNum = parseInt(episodeMatch[1], 10);
    const minutes = parseInt(timeMatch[1]);
    const seconds = parseInt(timeMatch[2]);
    const totalSeconds = minutes * 60 + seconds;
    
    const imgContainer = document.createElement('div');
    imgContainer.className = 'preview-frame-container';
    
    const placeholder = document.createElement('div');
    placeholder.className = 'preview-frame-placeholder';
    imgContainer.appendChild(placeholder);
    
    card.insertBefore(imgContainer, card.firstChild);
    
    try {
        const imageBlob = await extractFrame(episodeNum, totalSeconds, 'https://vv.noxylva.org');
        const imageUrl = URL.createObjectURL(imageBlob);
        
        const img = new Image();
        img.src = imageUrl;
        img.className = 'preview-frame';
        img.decoding = 'async';
        
        img.onload = () => {
            const originalCanvas = document.createElement('canvas');
            originalCanvas.width = img.width;
            originalCanvas.height = img.height;
            const originalCtx = originalCanvas.getContext('2d');
            originalCtx.drawImage(img, 0, 0);
            
            const displayCanvas = document.createElement('canvas');
            displayCanvas.width = img.width;
            displayCanvas.height = img.height;
            displayCanvas.className = 'preview-frame';
            
            displayCanvas.originalCanvas = originalCanvas;
            
            const renderCanvas = () => {
                const ctx = displayCanvas.getContext('2d');
                ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
                ctx.drawImage(originalCanvas, 0, 0);
                
                if (watermarkLoaded && AppState.showWatermark) {
                    const watermarkScale = displayCanvas.width * 0.25 / watermarkImage.width;
                    const watermarkWidth = watermarkImage.width * watermarkScale;
                    const watermarkHeight = watermarkImage.height * watermarkScale;
                    
                    ctx.drawImage(
                        watermarkImage,
                        displayCanvas.width - watermarkWidth - 5,
                        displayCanvas.height - watermarkHeight - 5,
                        watermarkWidth,
                        watermarkHeight
                    );
                }
            };
            
            renderCanvas();
            
            if (!window.canvasRenderQueue) {
                window.canvasRenderQueue = new Set();
            }
            window.canvasRenderQueue.add(displayCanvas);
            
            imgContainer.appendChild(displayCanvas);
            setTimeout(() => {
                displayCanvas.classList.add('loaded');
                placeholder.style.opacity = '0';
                setTimeout(() => placeholder.remove(), 300);
            }, 50);
            
            URL.revokeObjectURL(imageUrl);
        };

        img.onerror = () => {
            console.error('[Preview] Failed to load image:', imageUrl);
            imgContainer.remove();
            URL.revokeObjectURL(imageUrl);
        };
    } catch (error) {
        console.error('[Preview] Failed to extract frame:', error);
        imgContainer.remove();
    }
}

function getEpisodeUrl(filename) {
    for (let key in mapping) {
        if (mapping[key] === filename) {
            return key;
        }
    }
    return null;
}


function startLoadingBar() {
    const loadingBar = document.getElementById('loadingBar');
    loadingBar.style.width = "0%";
    loadingBar.style.display = "block";

    if (loadingBar.interval) {
        clearInterval(loadingBar.interval);
    }
    
    let progress = 0;
    loadingBar.interval = setInterval(() => {
        const currentWidth = parseFloat(loadingBar.style.width);
        if (currentWidth > progress + 1) {
            clearInterval(loadingBar.interval);
            return;
        }
        
        progress += 0.5;
        if (progress > 95) {
            clearInterval(loadingBar.interval);
            progress = 95;
        }
        loadingBar.style.width = `${progress}%`;
    }, 30);
}

function completeLoadingBar() {
    const loadingBar = document.getElementById('loadingBar');
    clearInterval(loadingBar.interval);
    
    loadingBar.style.transition = 'width 0.3s ease-out';
    loadingBar.style.width = "100%";
    
    setTimeout(() => {
        loadingBar.style.display = "none";
        loadingBar.style.transition = '';
        loadingBar.style.width = "0%";
    }, 300);
}


function initializeScrollListener() {
    if (window.currentObserver) {
        window.currentObserver.disconnect();
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && 
                AppState.hasMoreResults && 
                !AppState.isSearching && 
                AppState.cachedResults.length > AppState.displayedCount) {
                
                displayResults({
                    status: 'success',
                    data: AppState.cachedResults,
                    count: AppState.cachedResults.length
                }, true);
            }
        });
    }, {
        root: null,
        rootMargin: '200px',
        threshold: 0.1
    });

    window.currentObserver = observer;

    const oldTrigger = document.getElementById('scroll-trigger');
    if (oldTrigger) {
        oldTrigger.remove();
    }

    const trigger = document.createElement('div');
    trigger.id = 'scroll-trigger';
    trigger.style.cssText = 'height: 20px; margin: 20px 0;';
    document.getElementById('results').appendChild(trigger);
    
    observer.observe(trigger);
}

function handleCardClick(result) {
    const episodeMatch = result.filename.match(/\[P(\d+)\]/);
    const timeMatch = result.timestamp.match(/^(\d+)m(\d+)s$/);
    
    if (episodeMatch && timeMatch) {
        const episodeNum = parseInt(episodeMatch[1], 10);
        const minutes = parseInt(timeMatch[1]);
        const seconds = parseInt(timeMatch[2]);
        const totalSeconds = minutes * 60 + seconds;
        
        for (const [url, filename] of Object.entries(mapping)) {
            if (filename === result.filename) {
                const videoUrl = `https://www.bilibili.com${url}?t=${totalSeconds}`;
                window.open(videoUrl, '_blank');
                break;
            }
        }
    }
}