const output = document.getElementById('output')

function getDelayInMs() {
    return +document.getElementById('task-delay').value * 1000
}

/**
 * Wait for some amount of time
 * @param {number} time
 */
function wait(time) {
    // Randomize time
    const delta = time * 0.35
    const min = -delta
    const max = delta

    const randomTime = time + Math.random() * (max - min) + min

    // https://stackoverflow.com/a/39914235
    const promise = new Promise((resolve) => setTimeout(resolve, randomTime))

    return {
        promise,
        randomTime,
    }
}

async function runAction(scriptName) {
    // Run script from `actions` folder in a current tab
    const [result] = await browser.tabs.executeScript(undefined, {
        file: `/actions/${scriptName}`,
    })

    return result
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

    async start() {
        let fullIndex = []
        let currentPageInfo
        let count = 1

        this.#showStopButton()

        while (true) {
            if (this.#stopFlagRaised) {
                break
            }

            // Get info about current page
            currentPageInfo = await runAction('get_page_info.js')

            // Get current page index entries
            const currentIndex = await runAction('get_index.js')
            fullIndex.push(...currentIndex)

            // Go to a next index page, if it does exist
            if (currentPageInfo.nextPageUrl) {
                await changeURL(currentPageInfo.nextPageUrl)
                const { promise, randomTime } = wait(getDelayInMs())

                this.#statusEl.innerText = `Odwiedzone strony: ${count}\nCzas oczekiwania: ${Math.floor(randomTime / 1000)}s`
                await promise

                count += 1
            } else {
                break
            }
        }

        if (fullIndex.length > 0) {
            const filename = `${currentPageInfo.name}, ${currentPageInfo.sort}.json`
            await downloadObjectAsJSON(fullIndex, filename)
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

    async download() {
        const thread = await runAction('get_thread.js')

        const filename = `${thread.title}, ${thread.author}, ${thread.published}.json`
        await downloadObjectAsJSON(thread, filename)
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
            const { promise, randomTime } = wait(getDelayInMs())

            this.#statusEl.innerText = `Status pobierania: ${i}/${topRange}\nCzas oczekiwania: ${Math.floor(randomTime / 1000)}s`
            await promise

            // Download
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
