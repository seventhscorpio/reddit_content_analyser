const output = document.getElementById('output')

function getDelayInMs() {
    return +document.getElementById('task-delay').value * 1000
}

/**
 * Wait for some amount of time
 * @param {number} time
 * @param {() => boolean} earlyResolve
 */
function wait(time, earlyResolve) {
    // Randomize time
    const delta = time * 0.35
    const min = -delta
    const max = delta

    const randomTime = time + Math.random() * (max - min) + min

    // https://stackoverflow.com/a/39914235
    const promise = new Promise((resolve) => {
        let earlyResolveId = null

        // Schedule promise resolve
        const resolveId = setTimeout(() => {
            // Clear early resolve interval, if it was set
            if (earlyResolveId !== null) {
                clearInterval(earlyResolveId)
            }

            // Resolve promise at scheduled time
            resolve()
        }, randomTime)

        if (typeof earlyResolve === 'function') {
            // Run periodic checks for early wait resolve
            earlyResolveId = setInterval(() => {
                // Run passed callback
                const shouldResolveNow = earlyResolve()

                if (shouldResolveNow) {
                    // Cancel scheduled wait resolve
                    clearTimeout(resolveId)

                    // Resolve now
                    resolve()
                }
            }, 200)
        }
    })

    return {
        promise,
        randomTime,
    }
}

/**
 * @template TResult
 * @typedef {Object} ActionResult
 * @property {boolean} success
 * @property {TResult|null} result
 * @property {string|null} errorMessage
 */

/**
 * Inject script from `actions` folder into current tab and return value calculated by it
 * @template TResult
 * @see https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/executeScript#examples
 * @param {string} scriptName
 * @returns {Promise<ActionResult<TResult>>}
 */
async function runAction(scriptName) {
    try {
        // Try to run script from `actions` folder in a current tab
        const [result] = await browser.tabs.executeScript(undefined, {
            file: `/actions/${scriptName}`,
        })

        if (!result) {
            // Return error when result is empty
            return {
                success: false,
                result: null,
                errorMessage: `Action "${scriptName}" returned empty result`,
            }
        } else {
            // Return action result
            return {
                success: true,
                result,
                errorMessage: null,
            }
        }
    } catch (e) {
        // Return error message
        return {
            success: false,
            result: null,
            errorMessage: e,
        }
    }
}

async function changeURL(url) {
    console.log(url)

    await browser.tabs.update(undefined, {
        url,
    })
}

async function downloadObjectAsJSON(obj, filename) {
    const blob = new Blob([JSON.stringify(obj)], {
        type: 'application/json',
    })

    const url = URL.createObjectURL(blob)
    filename = filename.replace(/[^a-z0-9\.]/gi, '_')

    await browser.downloads.download({ url, filename })
}

/**
 *
 * @returns {Promise<File>}
 */
function pickFile() {
    const field = document.createElement('input')
    field.style.display = 'none'
    field.setAttribute('type', 'file')

    document.body.appendChild(field)
    field.click()

    return new Promise((resolve) => {
        field.addEventListener('change', (e) => {
            const file = e.target.files[0]
            document.body.removeChild(field)
            resolve(file)
        })
    })
}

/**
 *
 * @param {string} errorMessage
 */
function logError(errorMessage) {
    const listEl = document.getElementById('logs-list')

    const entryEl = document.createElement('li')
    const timeEl = document.createElement('time')
    const textEl = document.createElement('span')

    timeEl.innerText = new Date(Date.now()).toLocaleTimeString('pl-PL', {
        timeStyle: 'medium',
    })

    textEl.innerText = errorMessage

    entryEl.appendChild(timeEl)
    entryEl.appendChild(textEl)

    if (listEl.childNodes.length <= 0) {
        listEl.appendChild(entryEl)
    } else {
        listEl.insertBefore(entryEl, listEl.firstChild)
    }

    listEl.scrollIntoView()
}

/**
 * Return current time for a file name
 * @returns {string}
 */
function getCurrentTimeForFilename() {
    const time = new Date(Date.now())
        .toLocaleString('pl-PL', { dateStyle: 'short', timeStyle: 'short' })
        .replaceAll(':', '')
        .replaceAll('.', '')
        .replaceAll(', ', '_')

    return time
}

class GetFullIndexTask {
    #statusEl
    #startButtonEl
    #stopButtonEl

    #showStartButton() {
        this.#startButtonEl.style.display = 'block'
        this.#stopButtonEl.style.display = 'none'
    }

    #showStopButton() {
        this.#startButtonEl.style.display = 'none'
        this.#stopButtonEl.style.display = 'block'
    }

    constructor() {
        this.#statusEl = document.getElementById('index-status')

        this.#startButtonEl = document.getElementById(
            'task-get-full-index-start',
        )
        this.#stopButtonEl = document.getElementById('task-get-full-index-stop')
        this.#stopFlagRaised = false

        this.#startButtonEl.addEventListener('click', this.start.bind(this))
        this.#stopButtonEl.addEventListener('click', this.stop.bind(this))

        this.#showStartButton()
    }

    #stopFlagRaised

    /**
     * Start crawling through index page
     */
    async start() {
        let filename = null
        let fullIndex = []
        let count = 1

        this.#showStopButton()

        while (true) {
            // Exit on stop button press
            if (this.#stopFlagRaised) {
                break
            }

            // Try to get info about current page
            const pageInfo = await runAction('get_page_info.js')

            // If we can't determine page type, log error and stop crawling
            if (!pageInfo.success) {
                logError(`Błąd skryptu: ${pageInfo.errorMessage}`)
                break
            }

            // Check if we're on a index page. If not, stop crawling
            if (pageInfo.result.type !== 'index') {
                logError(
                    `Natrafiono na stronę, która nie jest indeksem: ${pageInfo.result.href}. Zbieranie indeksu zostanie zakończone`,
                )
                break
            }

            // Set index file name if not already set
            if (filename === null) {
                filename = `${pageInfo.result.name}_${pageInfo.result.sort}_${getCurrentTimeForFilename()}.json`
            }

            // Try to get current page index entries
            const index = await runAction('get_index.js')

            if (!index.success) {
                logError(
                    `Błąd skryptu: "${index.errorMessage}". Strona ${pageInfo.result.href} zostanie pominięta`,
                )
            } else {
                // Add found entries to collection
                fullIndex.push(...index.result)
            }

            // Go to a next index page, if it does exist
            if (pageInfo.result.nextPageUrl) {
                await changeURL(pageInfo.result.nextPageUrl)

                // Wait until random delay elapses or stop button is pressed
                const { promise, randomTime } = wait(
                    getDelayInMs(),
                    () => this.#stopFlagRaised,
                )

                this.#statusEl.innerText = `Odwiedzone strony: ${count}\nCzas oczekiwania: ${Math.floor(randomTime / 1000)}s`
                await promise

                count += 1
            } else {
                break
            }
        }

        if (fullIndex.length > 0) {
            await downloadObjectAsJSON(
                fullIndex,
                filename ||
                    `Unknown_subreddit_index_${getCurrentTimeForFilename()}.json`,
            )
        }

        this.#showStartButton()
    }

    async stop() {
        this.#stopFlagRaised = true
    }
}

class GetThreadTask {
    #statusEl
    #loadIndexButtonEl

    #rangeEl
    #rangeTopEl
    #rangeBottomEl

    #startButtonEl
    #stopButtonEl

    #downloadButtonEl

    #index

    #showStartButton() {
        this.#startButtonEl.style.display = 'block'
        this.#stopButtonEl.style.display = 'none'
    }

    #showStopButton() {
        this.#startButtonEl.style.display = 'none'
        this.#stopButtonEl.style.display = 'block'
    }

    #hideIndexUI() {
        this.#rangeEl.style.display = 'none'
        this.#startButtonEl.style.display = 'none'
        this.#stopButtonEl.style.display = 'none'
    }

    #showIndexUI() {
        this.#rangeEl.style.display = 'flex'
        this.#startButtonEl.style.display = 'block'
    }

    constructor() {
        this.#statusEl = document.getElementById('thread-status')
        this.#loadIndexButtonEl = document.getElementById('task-load-index')

        this.#rangeEl = document.getElementById('thread-index-range')
        this.#rangeTopEl = document.getElementById('thread-index-top-range')
        this.#rangeBottomEl = document.getElementById(
            'thread-index-bottom-range',
        )

        this.#startButtonEl = document.getElementById(
            'task-get-all-threads-start',
        )
        this.#stopButtonEl = document.getElementById(
            'task-get-all-threads-stop',
        )

        this.#downloadButtonEl = document.getElementById(
            'task-get-current-thread',
        )

        this.#loadIndexButtonEl.addEventListener(
            'click',
            this.loadIndex.bind(this),
        )
        this.#startButtonEl.addEventListener('click', this.start.bind(this))
        this.#stopButtonEl.addEventListener('click', this.stop.bind(this))
        this.#downloadButtonEl.addEventListener(
            'click',
            this.download.bind(this),
        )

        this.#stopFlagRaised = false

        this.#hideIndexUI()
    }

    /**
     * Try to download a thread
     * @returns {Promise<boolean>}
     */
    async download() {
        // Try to get info about current page
        const pageInfo = await runAction('get_page_info.js')

        // If we can't determine page type, log error and return false
        if (!pageInfo.success) {
            logError(`Błąd skryptu: ${pageInfo.errorMessage}`)
            return false
        }

        // Check if we're on a thread page
        if (pageInfo.result.type !== 'thread') {
            logError(`Strona nie jest wątkiem: ${pageInfo.result.href}`)
            return false
        }

        // Try to get thread content
        const thread = await runAction('get_thread.js')

        if (!thread.success) {
            logError(`Błąd skryptu: ${thread.errorMessage}`)
            return false
        } else {
            const { title, author, published } = thread.result
            const filename = `${title}_${author}_${published}_${getCurrentTimeForFilename()}.json`

            try {
                await downloadObjectAsJSON(thread.result, filename)
                return true
            } catch (e) {
                console.debug(e)

                logError(
                    `Wystąpił błąd podczas zapisywania wątku do pliku: ${thread.href}`,
                )
                return false
            }
        }
    }

    async loadIndex() {
        // Read index file
        const indexFile = await pickFile()
        const index = JSON.parse(await indexFile.text())

        // Update status text
        this.#statusEl.innerText = `Załadowano indeks z liczbą wątków: ${index.length}`

        this.#rangeBottomEl.value = 0
        this.#rangeTopEl.value = index.length - 1
        this.#rangeTopEl.max = index.length - 1

        this.#index = index
        this.#showIndexUI()
    }

    #stopFlagRaised

    async start() {
        const bottomRange = +this.#rangeBottomEl.value
        const topRange = +this.#rangeTopEl.value

        this.#stopFlagRaised = false
        this.#showStopButton()

        for (let i = bottomRange; i < topRange; ++i) {
            const summary = this.#index[i]

            // Go to thread URL
            await changeURL(summary.url)
            const { promise, randomTime } = wait(
                getDelayInMs(),
                () => this.#stopFlagRaised,
            )

            this.#statusEl.innerText = `Status pobierania: ${i}/${topRange}\nCzas oczekiwania: ${Math.floor(randomTime / 1000)}s`
            await promise

            // Try to download a thread
            await this.download()

            if (this.#stopFlagRaised) {
                break
            }
        }

        this.#showStartButton()
    }

    async stop() {
        this.#stopFlagRaised = true
    }
}

// Init tasks
new GetFullIndexTask()
new GetThreadTask()
