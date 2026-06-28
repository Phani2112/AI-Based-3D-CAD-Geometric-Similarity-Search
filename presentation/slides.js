const slides = Array.from(document.querySelectorAll('.slide'));
const progressBar = document.getElementById('progressBar');
const slideCounter = document.getElementById('slideCounter');
let currentSlide = 0;

function clampSlide(index) {
  return Math.max(0, Math.min(index, slides.length - 1));
}

function updateUrl(index) {
  const slideNumber = index + 1;
  if (window.location.hash !== `#${slideNumber}`) {
    window.history.replaceState(null, '', `#${slideNumber}`);
  }
}

function goToSlide(index) {
  currentSlide = clampSlide(index);
  slides.forEach((slide, slideIndex) => {
    slide.classList.toggle('active', slideIndex === currentSlide);
    slide.setAttribute('aria-hidden', slideIndex === currentSlide ? 'false' : 'true');
  });
  const slideNumber = currentSlide + 1;
  if (slideCounter) slideCounter.textContent = `${slideNumber} / ${slides.length}`;
  if (progressBar) progressBar.style.width = `${(slideNumber / slides.length) * 100}%`;
  updateUrl(currentSlide);
}

function nextSlide() {
  goToSlide(currentSlide + 1);
}

function previousSlide() {
  goToSlide(currentSlide - 1);
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {
    event.preventDefault();
    nextSlide();
  }
  if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
    event.preventDefault();
    previousSlide();
  }
  if (event.key === 'Home') {
    event.preventDefault();
    goToSlide(0);
  }
  if (event.key === 'End') {
    event.preventDefault();
    goToSlide(slides.length - 1);
  }
});

document.getElementById('nextSlide')?.addEventListener('click', nextSlide);
document.getElementById('prevSlide')?.addEventListener('click', previousSlide);

const hashSlide = Number.parseInt(window.location.hash.replace('#', ''), 10);
goToSlide(Number.isFinite(hashSlide) ? hashSlide - 1 : 0);
